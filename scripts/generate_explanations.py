#!/usr/bin/env python3
"""

How to use:

  # insert in the terminal
  python3 generate_explanations.py \
      --model deepseek-v4-pro:cloud \
      --dir "directory/path" \
      --outdir explanations

"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXPLAINER_TIMEOUT = 600   # cloud reasoning models are slow; give them room
OLLAMA_URL = "http://localhost:11434/api/generate"

# --------------------------------------------------------------------------
# THE FIVE CANDIDATE PROMPTS
#
# Held constant: the inputs ({finding}, {tf}) and the requested output (a prose
# explanation for an engineer). What varies is only the strategy.
# --------------------------------------------------------------------------

PROMPTS = {}

# -- P1: minimal role and task ---------------------------------------------
# The floor for prompting. A bare instruction with no strategy at all. Every
# other prompt has to beat this to justify its extra complexity.
PROMPTS["p1_minimal"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below.

Explain the problem to an engineer who has to fix it. Address every failing check shown in the scanner finding.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P0: neutral explainer-comparison prompt -------------------------------
# Verbatim copy of EXPLAIN_PROMPT from compare_explainers.py. Used only for
# the post-hoc explainer comparison, so that only the MODEL varies.
PROMPTS["p0_neutral"] = """You are a cloud security expert. A static analysis scanner (Checkov) flagged a \
security misconfiguration in the Terraform file below.

Write a clear, accurate explanation for an engineer covering:
1. What the misconfiguration is.
2. Why it matters (the concrete security risk / blast radius).
3. How to fix it safely.

Rules:
- Only claim things the file and finding actually support. Do NOT invent \
resources, attack scenarios, or impact that is not evidenced.
- Be concise and specific. No filler.

--- Checkov finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P2: scanner-output grounding ------------------------------------------
# Forces every claim back to the evidence. Tests whether disciplined grounding
# helps the fixer, or merely makes the explanation drier.
PROMPTS["p7_grounded"] = """You are a cloud security expert. A scanner has flagged a security misconfiguration in the Terraform file below.

Explain the problem to an engineer who has to fix it. Ground every claim in the evidence you have been given:

- Refer to the specific check ID and the specific resource it was raised against.
- Point to the exact lines in the file that trigger the check.
- State only what the finding and the file support. If something is not evidenced, do not assert it. If you are uncertain, say so.
- Do not invent resources, attack scenarios, or impact that you cannot point to in the given material.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P3: chain-of-thought ---------------------------------------------------
# Tests whether reasoning before writing produces a better explanation. Note the
# tension the Background raises: chain-of-thought can produce plausible-sounding
# reasoning that is not faithful. If that happens, it should show up here.
PROMPTS["p2_cot"] = """You are a cloud security expert. A scanner has flagged a security misconfiguration in the Terraform file below.

Work through the problem step by step before you explain it:

1. Identify which resource in the file each check was raised against.
2. Work out what that resource currently does, and what the check requires of it.
3. Explain the security consequence that the rule and this configuration support. Do not invent an attack path that the evidence does not establish.
4. Work out what change would satisfy the check while preserving the resource's evident purpose.

Then write your explanation for an engineer who has to fix it, informed by that reasoning. Carry out the steps above as part of your reasoning, but output only the final explanation, not the numbered working.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P4: rule-reference augmentation ----------------------------------------
# Provides the rule metadata Checkov carries for each finding: the check id, the
# check name, and the generic documentation link. This is reference augmentation
# rather than true retrieval-augmented generation, since the page contents are
# not fetched; the prompt receives the references the finding already carries,
# not retrieved knowledge. It is therefore a direct test of the RQ1 claim that
# the doc link is generic and adds little. If P4 does not beat P1, that is a
# result rather than a failure. Because it introduces information the other
# prompts do not receive, it is reported as a reference-augmented condition
# distinct from the prompt-only strategies.
PROMPTS["p8_ruleref"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below. Reference material for the rules that were triggered is also provided.

Explain the problem to an engineer who has to fix it. Address every failing check. Use the reference material to ground the general security rationale, and use the file to ground what is specific to this configuration. Where the reference is generic, say what it means for this particular file rather than repeating it. Treat the reference as general rule guidance; do not copy example configurations from it unless they are directly applicable to the supplied file.

--- Scanner finding ---
{finding}

--- Reference documentation ---
{docs}

--- Terraform file ---
{tf}
"""

# -- P5: guided remediation / playbook --------------------------------------
# The most action-oriented prompt. Expected to do well on fix rate. Watch the
# confound: if it wins by a wide margin, check whether it won by explaining or by
# handing over the answer. The has_hcl column in the manifest is there for that.
PROMPTS["p3_playbook"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below.

Write a remediation briefing for the engineer who has to fix them. Structure it as follows:

- Take each failing check in the order shown by the scanner finding.
- For each one: name the check and the resource, state in one line what is wrong, state why it matters, and describe in prose the change needed to make the check pass. Do not provide a complete Terraform block or a ready-to-paste replacement.
- Note interactions between fixes only where the supplied file supports them.

Cover every failing check. Do not leave any unaddressed.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P6: plan-and-solve decomposition ---------------------------------------
# Where P3 reasons freely, P6 forces an explicit plan first, then solves each
# sub-part in turn. The hypothesis is that decomposition helps most on the
# multi-check files, where a holistic pass tends to miss individual checks. This
# is Plan-and-Solve decomposition; it is not least-to-most, since it does not
# order sub-problems by complexity or feed earlier solutions into later ones.
PROMPTS["p4_plansolve"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below.

First devise a plan, then carry it out.

PLAN: List the sub-problems to be solved. For each failing check, write one line naming the check, the resource it was raised against, and what it requires.

SOLVE: Work through your plan one sub-problem at a time. For each, explain what is currently wrong in this file, why it matters, and describe in prose the change that would make the check pass. Handle each failing check separately and completely before moving to the next, so that none is skipped. Do not provide a complete Terraform block or a ready-to-paste replacement.

Develop the PLAN and SOLVE stages as part of your reasoning, but output only the final explanation for the engineer who has to fix the file, not the plan or the intermediate working.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P7: chain-of-verification ----------------------------------------------
# Draft, then interrogate the draft, then keep only what survives. Aimed
# squarely at the plausibility-faithfulness gap: it should strip out confident
# claims the evidence does not support.
PROMPTS["p9_verify"] = """You are a cloud security expert. A scanner has flagged a security misconfiguration in the Terraform file below.

Produce your explanation in three stages.

1. DRAFT: Write an initial explanation of the problem and its fix for an engineer.

2. VERIFY: Generate a set of verification questions that test the important claims in your draft. Answer each question directly from the scanner finding and the Terraform file, without relying on the wording or conclusions of the draft. Mark any question the evidence cannot answer.

3. REVISE: Rewrite the explanation using only the answers that the evidence supported. Drop or explicitly qualify anything that did not survive verification.

Carry out the DRAFT and VERIFY stages as part of your reasoning, but output only the final revised explanation, not the draft, the verification questions, or their answers.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P8: schema-constrained structured output -------------------------------
# Forces an evidence slot for every claim. Deliberately passed to the fixer in
# its native JSON form: since usefulness is measured by downstream repair, an
# explanation the fixer cannot act on is by that measure less useful, and
# rendering it to prose would measure a different explanation than the one this
# prompt produces. Whether a small fixer can consume structured input is part of
# what this condition tests.
PROMPTS["p11_schema"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below.

Respond with a single JSON object and nothing else. Include a top-level "findings" array with exactly one entry per failing check (including the same check id raised against different resources). Each entry must have exactly these fields and no others:

- "check_id": the rule identifier (string)
- "resource": the resource the check was raised against (string)
- "evidence": an array of one or more verbatim excerpts from the file that trigger the check
- "risk": what could go wrong, stated only from what the evidence supports (string)
- "remediation": the change that makes the check pass, described in words rather than as a ready-to-paste block (string)
- "confidence": one of "low", "medium", "high", reflecting how well the evidence supports your explanation (string)

Every field must be filled from the finding and the file; do not assert anything the evidence does not support. Return valid JSON using double quotes, with no comments, no trailing commas, no markdown fences, and no text before or after the object.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P9: constitutional / rubric-based self-critique -------------------------
# Turns this study's own M2 faithfulness criteria into an in-prompt control: the
# model writes, critiques against explicit principles, and revises. Tests
# whether making the evaluation rubric visible to the generator improves the
# explanation it produces.
PROMPTS["p10_constitutional"] = """You are a cloud security expert. A scanner has flagged a security misconfiguration in the Terraform file below.

Write an explanation for the engineer who has to fix it, then hold it to the following principles:

- Every risk claim must be explicitly tied to the triggered rule or a relevant configuration excerpt.
- No remediation may be given without pointing to what in the file it changes.
- Nothing may be asserted that the finding and the file do not support; uncertainty must be stated as uncertainty.
- The explanation must address every failing check, and must not introduce resources or facts that are not present in the evidence.

First write the explanation. Then perform one systematic critique against each principle in turn, noting any breach. Then produce one revised explanation that satisfies every principle. Carry out the draft and critique as part of your reasoning, but output only the final revised explanation.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P10: contrastive / counterfactual --------------------------------------
# Frames the explanation as the delta between the broken file and a secure one.
# For a repair task this is the most on-target framing available: it names
# exactly what must change. Not drawn from a single named technique, so it is a
# small original contribution rather than a re-run of a known method.
PROMPTS["p6_contrastive"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below.

Explain the problem by contrasting the current configuration with a configuration that would satisfy the triggered check.

For each failing check:
- Describe what the file currently does that makes the check fail.
- Describe what a configuration that satisfies the check would do instead.
- State precisely what differs between the two, holding the rest of the configuration constant, and why that specific difference is what matters for security.

Then give the engineer who has to fix the file a short explanation built from those contrasts, so that it is clear exactly what must change and why. Describe the difference at the level of the configuration property rather than supplying a complete Terraform replacement.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""

# -- P11: step-back / principle-first ---------------------------------------
# Where P3 and P6 reason within the instance, step-back prompting (Zheng et al.
# 2023) first abstracts to the general principle, then applies it to the concrete
# case. Checkov checks are concrete instances of general security principles, so
# this is a natural fit: name the invariant the rule enforces, then explain how
# this file violates it. The risk is a drift into generic security advice, so the
# final step forces the explanation back onto the evidence in the supplied file.
PROMPTS["p5_stepback"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below.

Approach the problem in two stages.

STEP BACK: For each failing check, first state the general security principle the check exists to enforce (for example, encryption at rest, least privilege, or restricting public exposure), independently of this particular file.

APPLY: Then apply that principle to the specific resource in this file. Explain, using only the evidence in the finding and the file, exactly how this configuration violates the principle, why that matters here, and what change would bring it into line.

Carry out the step-back stage as part of your reasoning, but output only the final explanation for the engineer who has to fix the file. Ground every claim in the supplied finding and file; do not offer generic security advice that the evidence does not support.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""


# --------------------------------------------------------------------------
# checkov  (identical to the scan() in fullfix_conditions.py, so the finding the
# explainer sees is exactly the finding Condition A feeds the fixer)
# --------------------------------------------------------------------------
def scan(path):
    """Run Checkov. Returns (failed_count, failed_summary, check_ids, parse_ok)."""
    try:
        out = subprocess.run(
            ["checkov", "-f", path, "--compact", "--quiet", "-o", "json"],
            capture_output=True, text=True, timeout=300,
        )
        data = json.loads(out.stdout)
        if isinstance(data, list):
            data = data[0] if data else {}
        results = data.get("results", {})
        if not results:
            return -1, "file could not be parsed by Checkov", [], False
        failed = results.get("failed_checks", [])
        def _fmt(c):
            rng = c.get("file_line_range") or []
            loc = f" (lines {rng[0]}-{rng[1]})" if len(rng) == 2 else ""
            return f"- {c.get('check_id')} on {c.get('resource')}{loc}: {c.get('check_name')}"
        summary = "\n".join(_fmt(c) for c in failed)
        # keep the check metadata for the RAG prompt: id, name, and doc link
        checks = []
        seen = set()
        for c in failed:
            cid = c.get("check_id")
            if cid in seen:
                continue
            seen.add(cid)
            checks.append({
                "id": cid,
                "name": c.get("check_name"),
                "guideline": c.get("guideline") or "",
            })
        return len(failed), summary, checks, True
    except FileNotFoundError:
        sys.exit("ERROR: `checkov` not found on PATH. Activate the env you scan from.")
    except Exception as e:
        return -1, f"checkov error: {e}", [], False


def build_docs(checks):
    """Assemble the 'retrieved' reference documentation for the RAG prompt.

    Checkov attaches a guideline URL to each failed check. That link is the
    documentation an engineer is pointed at today, so it is the honest thing to
    retrieve: no external corpus is invented, and the prompt tests exactly what
    the scanner already offers.
    """
    if not checks:
        return "(no reference documentation available for these checks)"
    lines = []
    for c in checks:
        lines.append(f"{c['id']}: {c['name']}")
        if c["guideline"]:
            lines.append(f"  Reference: {c['guideline']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# explainer call
# --------------------------------------------------------------------------
def call_explainer(model, prompt, timeout=EXPLAINER_TIMEOUT):
    """Send the prompt to the explainer and return the explanation.

    This talks to Ollama's HTTP API rather than shelling out to `ollama run`.
    That is not a stylistic choice. When Ollama streams a reply to a terminal it
    redraws lines as they wrap, and those redraws are captured along with the
    text: words end up duplicated at line boundaries, so a captured explanation
    contains fragments like "Both gra" followed by "grant broad permissions".
    The text is corrupted in a way that is invisible until read closely, and an
    explanation corrupted in that way would be fed to the fixer and silently
    measured as if it were the model's actual output.

    The API returns the completion as a JSON string with no cursor control and
    no wrapping, so the text that arrives is the text the model produced. This
    is the same lesson as the escape-code problem in the repair pipeline: the
    result cannot be trusted until what is being measured is clean.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return None, f"cannot reach Ollama at {OLLAMA_URL}: {e}"
    except TimeoutError:
        return None, f"timed out after {timeout}s"
    except Exception as e:
        return None, f"api error: {e}"

    text = (data.get("response") or "").strip()

    # Thinking models emit a reasoning block before the answer. Strip it so the
    # explanation the fixer receives is the explanation and not the deliberation.
    if "...done thinking." in text:
        text = text.split("...done thinking.", 1)[1]
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = text.strip()

    if not text:
        return None, "empty output"
    return text, ""


HCL_BLOCK = re.compile(r'(?m)^\s*(?:resource|data|provider|module|variable)\s+"')
# A partial fix does not need a full block: a single assignment such as
# `encrypted = true` or a settings sub-block pasted into the explanation can hand
# the fixer the answer just as effectively. This catches attribute-level leakage
# that HCL_BLOCK misses: an identifier followed by `=` and a value, or a known
# remediation sub-block header. Recorded, not prevented, for the same reason as
# HCL_BLOCK: it turns the code-injection confound into a measured number.
HCL_ATTR = re.compile(r'(?m)^\s*[a-z][a-z0-9_]*\s*=\s*\S')
HCL_SUBBLOCK = re.compile(r'(?m)^\s*(?:server_side_encryption[a-z_]*|logging|versioning|'
                          r'encryption[a-z_]*|public_access_block|ingress|egress)\b')


def has_hcl(text):
    """Does the explanation contain a Terraform block?

    Recorded rather than prevented. An explanation that hands over the fix may
    raise the fix rate without having explained anything, which is the confound
    the Interpreting section flags. Logging it turns that worry into a number.

    A full block is the clearest case, but a partial fix (a single attribute
    assignment, or a pasted settings sub-block) hands over just as much, so those
    are caught too. For the schema (JSON) prompt, remediation values live inside
    JSON strings and will register here; that is expected, and the has_hcl figure
    for that prompt is therefore not directly comparable with the prose prompts.
    """
    return bool(HCL_BLOCK.search(text) or HCL_ATTR.search(text) or HCL_SUBBLOCK.search(text))


def json_validity(pname, text):
    """For the schema prompt, record whether the output is valid JSON. Empty for
    every other prompt. This is logged rather than enforced: whether a small
    fixer can consume structured input is part of what the schema condition
    tests, so the raw output is passed through unchanged, but its validity is
    recorded so the schema prompt's results can be read in that light."""
    if pname != "p11_schema":
        return ""
    stripped = text.strip()
    # tolerate a stray markdown fence if the model added one despite instructions
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", stripped).strip()
    try:
        json.loads(stripped)
        return "True"
    except Exception:
        return "False"


def wrap_corruption(text):
    """Count line boundaries where a word appears to have been duplicated.

    A guard against the terminal-wrapping corruption that the HTTP API is meant
    to avoid. The signature is a line that ends mid-word, followed by a line that
    begins with the completed form of that same word ("Both gra" / "grant broad
    permissions"). Clean output does not do this. If this count is anything other
    than zero, the explanations are not safe to feed to the fixer, and the run
    should be stopped rather than left to produce a result that measures
    corruption instead of explanation quality.

    Two refinements keep this from firing on legitimate text. Lines inside a code
    block are skipped, because adjacent lines of Terraform routinely share a
    prefix ("lambda:CreateFunction" followed by "lambda:CreateEventSourceMapping")
    without anything being wrong. And the duplicated fragment must be a genuine
    prefix of the following word rather than merely resembling it, since the real
    corruption always breaks a word in half and then repeats the whole of it.
    """
    lines = text.split("\n")
    hits = 0
    in_code = False
    for i in range(1, len(lines)):
        # track fenced code blocks and skip them
        if lines[i - 1].strip().startswith("```"):
            in_code = not in_code
        if in_code:
            continue

        prev, cur = lines[i - 1].strip(), lines[i].strip()
        if not prev or not cur:
            continue
        # skip anything that looks like code or markup rather than prose
        if any(c in prev for c in '{}=[]|') or any(c in cur for c in '{}=[]|'):
            continue
        if prev.startswith(("#", "-", "*", ">", "`")) or cur.startswith(("#", "-", "*", ">", "`")):
            continue

        pw = prev.split()[-1] if prev.split() else ""
        cw = cur.split()[0] if cur.split() else ""
        pw_clean = pw.strip('.,;:"\'()')
        cw_clean = cw.strip('.,;:"\'()')
        # the corruption repeats the broken word in full: the fragment left at the
        # end of one line is a strict prefix of the word that opens the next
        if (pw_clean and cw_clean and len(pw_clean) > 2
                and cw_clean != pw_clean
                and len(cw_clean) > len(pw_clean)
                and cw_clean.startswith(pw_clean)):
            hits += 1
    return hits


# --------------------------------------------------------------------------
PROMPTS["p12_combined"] = """You are a cloud security expert. A scanner has flagged one or more security misconfigurations in the Terraform file below.

Write a remediation briefing for the engineer who has to fix them, grounding every claim in the evidence you have been given.

Take each failing check in the order shown by the scanner finding. For each one:
- Name the check ID and the specific resource it was raised against.
- Point to the exact lines in the file that trigger it.
- State in one line what is wrong, and why it matters, using only what the finding and the file support. If something is not evidenced, do not assert it. If you are uncertain, say so.
- Describe in prose the change needed to make the check pass. Do not provide a complete Terraform block or a ready-to-paste replacement.

Cover every failing check. Do not leave any unaddressed. Do not invent resources, attack scenarios, or impact that you cannot point to in the given material.

--- Scanner finding ---
{finding}

--- Terraform file ---
{tf}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the explainer, e.g. deepseek-v4-pro:cloud")
    ap.add_argument("--dir", required=True, help="folder of .tf files (the dev set)")
    ap.add_argument("--outdir", default="explanations")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files")
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS.keys()),
                    help=f"which prompts to run (default all). Choices: {' '.join(PROMPTS)}")
    args = ap.parse_args()

    for p in args.prompts:
        if p not in PROMPTS:
            sys.exit(f"unknown prompt '{p}'. Choices: {' '.join(PROMPTS)}")

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".tf"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no .tf files found in {args.dir}")

    os.makedirs(args.outdir, exist_ok=True)
    for p in args.prompts:
        os.makedirs(os.path.join(args.outdir, p), exist_ok=True)

    total = len(args.prompts) * len(files)
    print(f"explainer = {args.model}")
    print(f"prompts   = {', '.join(args.prompts)}")
    print(f"files     = {len(files)}")
    print(f"total explanations to generate = {total}\n")

    # scan each file ONCE; every prompt then sees the identical finding, so the
    # comparison between prompts is not contaminated by scanner variation
    print("scanning files with Checkov...")
    findings = {}
    for fname in files:
        path = os.path.join(args.dir, fname)
        n, summary, checks, ok = scan(path)
        with open(path) as fh:
            tf = fh.read()
        findings[fname] = dict(n=n, summary=summary, checks=checks, tf=tf, ok=ok)
        if n <= 0:
            print(f"  WARNING: {fname} produced no findings (n={n}); it will be skipped")
    usable = [f for f in files if findings[f]["n"] > 0]
    print(f"  {len(usable)}/{len(files)} files have findings\n")

    manifest_path = os.path.join(args.outdir, "manifest.csv")
    mf = open(manifest_path, "w", newline="")
    writer = csv.DictWriter(mf, fieldnames=[
        "prompt", "file", "n_findings", "chars", "has_hcl", "json_valid",
        "corruption", "seconds", "error"])
    writer.writeheader()

    done = 0
    for pname in args.prompts:
        template = PROMPTS[pname]
        outsub = os.path.join(args.outdir, pname)
        print(f"=== {pname} ===")
        for fname in usable:
            d = findings[fname]
            done += 1
            fields = dict(finding=d["summary"], tf=d["tf"])
            if "{docs}" in template:
                fields["docs"] = build_docs(d["checks"])
            prompt = template.format(**fields)

            t0 = time.time()
            text, err = call_explainer(args.model, prompt)
            secs = time.time() - t0

            if text is None:
                print(f"  [{done}/{total}] {fname[:44]:44s} FAILED ({err})")
                writer.writerow(dict(prompt=pname, file=fname, n_findings=d["n"],
                                     chars=0, has_hcl="", json_valid="",
                                     corruption="",
                                     seconds=round(secs, 1), error=err))
                mf.flush()
                continue

            # the .txt suffix on the full filename is what Condition B looks for
            with open(os.path.join(outsub, fname + ".txt"), "w") as fh:
                fh.write(text + "\n")

            hcl = has_hcl(text)
            jv = json_validity(pname, text)
            corrupt = wrap_corruption(text)
            writer.writerow(dict(prompt=pname, file=fname, n_findings=d["n"],
                                 chars=len(text), has_hcl=hcl, json_valid=jv,
                                 corruption=corrupt,
                                 seconds=round(secs, 1), error=""))
            mf.flush()

            # A corrupted explanation is worse than a missing one, because it is
            # silently wrong. Stop the run rather than generate 150 of them.
            if corrupt > 0:
                print(f"\n  STOPPING: {fname} shows {corrupt} wrap-duplications.")
                print("  The captured text is corrupted, not the model's actual output.")
                print("  Explanations in this state would be fed to the fixer and measured")
                print("  as if they were clean. Fix the capture before continuing.")
                mf.close()
                sys.exit(1)

            badge = "HCL" if hcl else "   "
            print(f"  [{done}/{total}] {fname[:44]:44s} {len(text):6d} chars  {badge}  {secs:5.1f}s")

    mf.close()

    # ---------------------------------------------------------------- summary
    print("\n=== generation complete ===")
    rows = list(csv.DictReader(open(manifest_path)))
    for pname in args.prompts:
        pr = [r for r in rows if r["prompt"] == pname]
        ok = [r for r in pr if not r["error"]]
        if not ok:
            print(f"  {pname:12s}  0 explanations (all failed)")
            continue
        chars = [int(r["chars"]) for r in ok]
        hcl_n = sum(1 for r in ok if r["has_hcl"] == "True")
        corr = sum(int(r["corruption"] or 0) for r in ok)
        print(f"  {pname:12s}  {len(ok):2d} explanations   "
              f"mean {sum(chars)//len(chars):5d} chars   "
              f"{hcl_n:2d}/{len(ok)} contain HCL   "
              f"corruption: {corr}")
    fails = [r for r in rows if r["error"]]
    if fails:
        print(f"\n  {len(fails)} generation(s) FAILED; those files will be skipped in Condition B")

    print(f"\nmanifest: {manifest_path}")
    print("\nNext: run Condition B once per prompt, e.g.")
    for pname in args.prompts:
        print(f"  python3 fullfix_conditions.py --model codegemma:7b \\")
        print(f"      --dir \"{args.dir}\" --condition B \\")
        print(f"      --explanations {os.path.join(args.outdir, pname)} \\")
        print(f"      --out codegemma_B_{pname}.csv")


if __name__ == "__main__":
    main()
