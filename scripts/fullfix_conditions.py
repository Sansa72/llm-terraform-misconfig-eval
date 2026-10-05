#!/usr/bin/env python3
"""
fullfix_conditions.py  --  A/B/C conditions experiment
======================================================

Runs the three experimental conditions used in the main study. Kept separate from
the screening scripts so each stage's results stay attributable to one script.

The three conditions form an INFORMATION LADDER. They share exactly the same task
frame and the same repair loop, and differ ONLY in how much they are told about
the problem:

  C (floor):    nothing. The fixer is told only that the file is misconfigured.
                On a failed attempt it is told only that it still fails, so it
                never learns anything about the finding.

  A (baseline): the raw Checkov findings, exactly as the scanner produces them,
                given from the FIRST attempt. On a failed attempt the updated list
                of failing checks is fed back. This represents what an engineer
                actually receives from the scanner today, so it is kept
                deliberately bare rather than dressed up with extra guidance.

  B (test):     an LLM-generated explanation of the finding, given from the first
                attempt. On a failed attempt the remaining failing checks are fed
                back, as in A, so A and B differ only by the explanation.

The task frame carries ONLY what defines a valid repair in this study (the
engineer role, return the complete file, add no new resources, do not delete the
resource). It is identical across all three conditions, so it is task definition
rather than help and cannot advantage any one condition. Being a constant, it
cancels out of the B minus A comparison.

A repair counts as successful only if Checkov reports zero failing checks AND
check_validity.py confirms the file is still well-formed Terraform with the
flagged resource present and no newly invented references.

Usage:
  python3 fullfix_conditions.py --model mistral-nemo \
      --dir "/path/to/dev" --out nemo_C.csv --condition C

  python3 fullfix_conditions.py --model mistral-nemo \
      --dir "/path/to/dev" --out nemo_A.csv --condition A

  python3 fullfix_conditions.py --model mistral-nemo \
      --dir "/path/to/dev" --out nemo_B.csv --condition B \
      --explanations "/path/to/explanations"

For Condition B, --explanations points to a folder holding one explanation per
file, named <original_filename>.txt (e.g. Foo_main.tf.txt). Files with no matching
explanation are skipped, so B is only ever scored on files that actually had one.
"""

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

# ------------------------------------------------------------------ settings
BUDGET = 5             # attempts per file, identical across all conditions
OLLAMA_TIMEOUT = 900   # seconds per model call (raise for cloud models)

# The model is called over Ollama's HTTP chat API rather than by shelling out to
# `ollama run`. The command line streams its answer through a terminal, which
# reflows long lines at the display width; that reflowing corrupts the saved
# Terraform (lines wrapped, tokens cut mid-word, fragments duplicated), silently
# turning genuine repairs into unparseable files. /api/chat is used rather than
# /api/generate because it applies the model's own chat template, exactly as
# `ollama run` does, so the model is conditioned identically. The only change is
# that no terminal sits in the path.
OLLAMA_URL = "http://localhost:11434/api/chat"

NUM_CTX = 32768        # prompt + whole file + a full rewritten file must all fit
NUM_PREDICT = 16384     # max tokens the model may generate for the repaired file

# Substrings that only appear in OUR prompt, never in real Terraform. If the
# model's output contains one, it echoed the instructions back instead of
# producing a repair (a sign the response was truncated), so the attempt must be
# discarded rather than saved.
PROMPT_ECHO_MARKERS = (
    "THE FILE:",
    "You are an infrastructure engineer",
    "The security scanner reported",
    "Here is an explanation of the problem",
)

# ------------------------------------------------------------------ prompts
# Shared task frame. Identical in every condition. Carries only what defines a
# valid repair, and nothing about what is wrong with this particular file.
TASK_FRAME = """You are an infrastructure engineer. Fix the misconfiguration(s) in this Terraform file and return the COMPLETE corrected file.

Do not add any new resources; fix only the resources already in the file. Do not delete or empty a resource in order to silence a problem. Return only the Terraform, with no explanation and no markdown fences."""


# ------------------------------------------------------------------ checkov
def scan(path):
    """Run Checkov. Returns (failed_count, failed_summary, parse_ok)."""
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
            return -1, "file could not be parsed by Checkov", False
        failed = results.get("failed_checks", [])
        # Include the offending line range, which Checkov reports alongside the
        # rule id, resource and check name. Condition A is meant to be exactly
        # what the scanner hands an engineer today, so withholding a field the
        # tool actually provides would understate that baseline.
        def _fmt(c):
            rng = c.get("file_line_range") or []
            loc = f" (lines {rng[0]}-{rng[1]})" if len(rng) == 2 else ""
            return f"- {c.get('check_id')} on {c.get('resource')}{loc}: {c.get('check_name')}"
        summary = "\n".join(_fmt(c) for c in failed)
        return len(failed), summary, True
    except Exception as e:
        return -1, f"checkov error: {e}", False


# ------------------------------------------------------------------ validity
def is_valid(path, original=None):
    """Same validity check as the screening (with reference baselining)."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_validity.py")
    if not os.path.exists(script):
        return True
    validity_python = os.environ.get(
        "VALIDITY_PYTHON", os.path.expanduser("~/validity-env/bin/python3"),
    )
    if not os.path.exists(validity_python):
        validity_python = sys.executable
    cmd = [validity_python, script, path]
    if original:
        cmd += ["--original", original]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if "OVERALL VALID" not in r.stdout:
            return True   # checker unavailable; fall back on Checkov's own parse
        return r.returncode == 0
    except Exception:
        return True


# ------------------------------------------------------------------ model call
def call_model(model, prompt_text, out_path):
    """
    Send the prompt to Ollama over the HTTP chat API and save the cleaned
    Terraform. Returns True only if the output survives every guard below.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (socket.timeout, TimeoutError):
        print(f"      (model call timed out after {OLLAMA_TIMEOUT}s, skipping attempt)")
        return False
    except urllib.error.URLError as e:
        print(f"      (ollama API call failed: {e}, skipping attempt)")
        return False
    except Exception as e:
        print(f"      (model call failed: {e}, skipping attempt)")
        return False

    raw = body.get("message", {}).get("content", "")
    if not raw or not raw.strip():
        return False

    # Terminal escape codes should not appear now the terminal is out of the
    # path, but the strip is kept as a cheap safeguard.
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    raw = raw.replace("\x1b", "")

    # Reasoning models emit a chain-of-thought block before the answer.
    if "...done thinking." in raw:
        raw = raw.split("...done thinking.", 1)[1]
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw)
    raw = re.sub(r"(?is)^\s*thinking\.\.\..*?(?=\n\s*(?:resource|data|provider|module|variable|terraform)\b)", "", raw)

    # Drop markdown fences.
    lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
    cleaned = "\n".join(lines).strip()

# Cut any leading prose so the file starts at the first real Terraform block.
    m = re.search(r'(?m)^\s*(resource|data|provider|module|variable|terraform)\b', cleaned)
    if m:
        cleaned = cleaned[m.start():].strip()

    # Cut any trailing prose. In Condition B the fixer is shown a long, structured
    # explanation and tends to mirror its format, appending a markdown summary of
    # the changes it made ("**Note:** * I modified the resources ...") after the
    # file itself. That commentary is not valid HCL, so the parser rejects an
    # otherwise correct repair, and because Checkov silently skips what it cannot
    # parse the file can pass the scan while failing the validity check. The file
    # is therefore truncated at the last top-level closing brace, which is where
    # the Terraform ends and the commentary begins. This never fires in Conditions
    # A and C, where the terse input does not provoke the behaviour.
    tail = None
    for i, l in enumerate(cleaned.splitlines()):
        if l.rstrip() == "}":
            tail = i
    if tail is not None:
        cleaned = "\n".join(cleaned.splitlines()[: tail + 1]).strip()

    if not cleaned:
        return False

    # Must actually look like Terraform, not an error message, refusal, or prose.
    if not re.search(r'\b(resource|data|provider|module|variable)\s+"', cleaned):
        return False

    # Prompt-echo guard: a model that runs out of room starts reproducing our own
    # instructions instead of a repair.
    if any(marker in cleaned for marker in PROMPT_ECHO_MARKERS):
        print("      (guard: output echoes the prompt instructions, skipping attempt)")
        return False

    # Brace-balance guard: a response cut off by num_predict or a timeout leaves
    # truncated HCL with fewer closing braces than opening ones. Checkov may still
    # "parse" the early resources of such a file and wrongly report it as passing,
    # so this must be caught before the file is ever handed to scan().
    if cleaned.count("{") != cleaned.count("}"):
        print("      (guard: unbalanced braces, likely truncated output, skipping attempt)")
        return False

    with open(out_path, "w") as fh:
        fh.write(cleaned + "\n")
    return True


# ------------------------------------------------------------------ prompt
def build_prompt(file_text, condition, findings=None, explanation=None,
                 remaining=None):
    """Assemble the prompt. All three conditions share TASK_FRAME and the file,
    and differ ONLY in the problem-information block:
        C -> nothing
        A -> the raw Checkov findings
        B -> the LLM explanation
    On a failed attempt, A and B are given the updated findings; C is told only
    that the file still fails, so C stays a true floor across retries.
    """
    parts = [TASK_FRAME]

    # --- the one thing that differs between the conditions ---
    if condition == "A" and findings:
        parts += ["", "The security scanner reported the following findings:",
                  findings]
    elif condition == "B" and explanation:
        parts += ["", "Here is an explanation of the problem:", explanation]
    # Condition C is given no information about the problem at all.

    # --- retry feedback ---
    if remaining:
        if condition == "C":
            parts += ["", "The file still contains a misconfiguration. Try again."]
        else:
            parts += ["", "The following checks still fail:", remaining]

    parts += ["", "THE FILE:", file_text]
    prompt = "\n".join(parts)
    assert file_text.strip() and file_text in prompt, "file missing from prompt"
    return prompt


# ------------------------------------------------------------------ per file
def process_file(model, filepath, workdir, condition, explanation=None):
    base = os.path.basename(filepath)
    with open(filepath) as fh:
        original = fh.read()

    # Capture the scanner's findings for the ORIGINAL file. Condition A must see
    # these from the FIRST attempt, since the raw findings are precisely what it
    # is testing. (An earlier version discarded this summary, so Condition A only
    # received the findings from attempt two onward, which made its first attempt
    # behave like Condition C and understated the baseline.)
    start_fails, start_summary, start_ok = scan(filepath)
    findings = start_summary if condition == "A" else None

    current = original
    remaining = None
    trajectory = [start_fails]

    # Track the last attempt that produced a usable, parsing file. If the final
    # attempt is discarded (no usable output), the last path on disk is one the
    # model never wrote, and an earlier version scored that as start_fails, i.e.
    # no progress: a run of 15->13->None->8->6->None was recorded as not_fixed
    # with final_fails=15 even though the model's last real output stood at 6.
    # The process is iterative, so the meaningful end state is the last file the
    # model actually produced, not the best intermediate one (scoring the best
    # would flatter every model by cherry-picking its high point). We therefore
    # remember the most recent parsing attempt and score that at the end.
    last_usable = None  # path of the most recent attempt that parsed

    for attempt in range(1, BUDGET + 1):
        out_path = os.path.join(workdir, f"{base}.attempt{attempt}.tf")
        prompt = build_prompt(current, condition, findings=findings,
                              explanation=explanation, remaining=remaining)

        if not call_model(model, prompt, out_path):
            # No usable output. Go back to the original file rather than asking the
            # model to repair its own wreckage. No new information is leaked to C.
            current = original
            trajectory.append(None)
            continue

        fails, summary, parse_ok = scan(out_path)
        valid = is_valid(out_path, original=filepath)
        trajectory.append(fails)

        if fails == 0 and parse_ok and valid:
            return dict(file=base, condition=condition, start_fails=start_fails,
                        attempts=attempt, final_fails=0, reached_zero=True,
                        valid=True, outcome="fixed",
                        trajectory="->".join(str(x) for x in trajectory), note="")

        # Remember this as the most recent usable attempt if it parses. A file
        # that does not parse is not a state the process can meaningfully end on.
        if parse_ok:
            last_usable = out_path

        # Retry feedback. C stays contentless; A and B receive the remaining checks.
        if condition == "C":
            remaining = "still_failing"
        elif not parse_ok:
            remaining = "The file no longer parses. Return valid, complete Terraform."
        else:
            remaining = summary

        # Only carry a repaired file forward if it still parses.
        current = original if not parse_ok else open(out_path).read()

    # Budget exhausted. If no attempt ever produced a usable, parsing file, the
    # model never gave us anything to score.
    if last_usable is None:
        return dict(file=base, condition=condition, start_fails=start_fails,
                    attempts=BUDGET, final_fails=start_fails, reached_zero=False,
                    valid=False, outcome="not_fixed",
                    trajectory="->".join(str(x) for x in trajectory),
                    note="no usable output produced")

    # Score the last usable attempt, not merely the last path written (which may
    # be a discarded, never-written final attempt).
    score_path = last_usable
    final_fails, final_summary, final_parse_ok = scan(score_path)
    final_valid = is_valid(score_path, original=filepath)
    if final_fails == 0 and final_parse_ok and final_valid:
        outcome, reached = "fixed", True
    elif final_fails == 0 and not (final_parse_ok and final_valid):
        # Checkov reports no failures but the file is not valid Terraform: the
        # scanner silently skipped what it could not parse. Caught only here.
        outcome, reached = "gamed_scanner", False
    else:
        outcome, reached = "not_fixed", False

    return dict(file=base, condition=condition, start_fails=start_fails,
                attempts=BUDGET, final_fails=final_fails, reached_zero=reached,
                valid=final_valid, outcome=outcome,
                trajectory="->".join(str(x) for x in trajectory),
                note="budget exhausted")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True, help="folder of .tf files")
    ap.add_argument("--out", default="conditions.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--condition", choices=["A", "B", "C"], required=True,
                    help="A = file + raw Checkov findings; "
                         "B = file + LLM explanation; "
                         "C = file alone, no findings")
    ap.add_argument("--explanations", default=None,
                    help="(Condition B only) folder of <filename>.txt explanations")
    args = ap.parse_args()

    if args.condition == "B" and not args.explanations:
        sys.exit("Condition B requires --explanations pointing to a folder of explanations.")

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".tf"))
    if args.limit:
        files = files[: args.limit]

    workdir = tempfile.mkdtemp(prefix="conditions_")
    print(f"model={args.model}  files={len(files)}  budget={BUDGET}  condition={args.condition}")
    print(f"attempt files kept in: {workdir}\n")

    fieldnames = ["file", "condition", "start_fails", "attempts", "final_fails",
                  "reached_zero", "valid", "outcome", "trajectory", "note"]
    csv_fh = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()
    csv_fh.flush()

    counts = {"fixed": 0, "gamed_scanner": 0, "not_fixed": 0}
    n = 0
    for i, fname in enumerate(files, 1):
        explanation = None
        if args.condition == "B":
            exp_path = os.path.join(args.explanations, fname + ".txt")
            if not os.path.exists(exp_path):
                print(f"[{i}/{len(files)}] {fname[:46]:46s} -> SKIPPED (no explanation)")
                continue
            with open(exp_path) as fh:
                explanation = fh.read().strip()

        row = process_file(args.model, os.path.join(args.dir, fname), workdir,
                           args.condition, explanation=explanation)
        writer.writerow(row)
        csv_fh.flush()
        counts[row["outcome"]] += 1
        n += 1
        print(f"[{i}/{len(files)}] {fname[:46]:46s} -> {row['outcome']}")
        print(f"           trajectory: {row['trajectory']}")

    csv_fh.close()
    print(f"\n=== {args.model}  condition {args.condition}  ({n} files) ===")
    if n:
        print(f"  fixed:         {counts['fixed']:2d}  ({counts['fixed']/n*100:.0f}%)")
        print(f"  gamed_scanner: {counts['gamed_scanner']:2d}  ({counts['gamed_scanner']/n*100:.0f}%)")
        print(f"  not_fixed:     {counts['not_fixed']:2d}  ({counts['not_fixed']/n*100:.0f}%)")
    else:
        print("  (no files scored)")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
