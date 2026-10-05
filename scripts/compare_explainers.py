#!/usr/bin/env python3
"""

INFORMAL gut-check for explainer model, not a formal experiment.

"""

import argparse
import json
import os
import subprocess
import sys
import textwrap

# --- CLOUD models can be slow (thinking + network). Give them room. ----------
EXPLAIN_TIMEOUT = 300  # seconds per explainer call

# --- The prompt used for the informal test -----------------------------------
# One neutral, single-pass "explain this finding" prompt. We are NOT testing
# prompt styles here (that is the later formal step); we are only comparing
# which MODEL writes a better explanation given the same instruction.
EXPLAIN_PROMPT = """You are a cloud security expert. A static analysis scanner (Checkov) flagged a \
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


# ------------------------------------------------------------------ checkov
# Copied verbatim from your fullfix_conditions.py so the finding text the
# explainer sees is IDENTICAL to what Condition A/B will feed later.
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
        summary = "\n".join(
            f"- {c.get('check_id')} on {c.get('resource')}: {c.get('check_name')}"
            for c in failed
        )
        return len(failed), summary, True
    except FileNotFoundError:
        sys.exit("ERROR: `checkov` not found on PATH. Activate the env you use for scanning.")
    except Exception as e:
        return -1, f"checkov error: {e}", False


# ------------------------------------------------------------------ model call
def call_model(model, prompt, timeout=EXPLAIN_TIMEOUT):
    """Send a prompt to an Ollama model and return its text output."""
    try:
        out = subprocess.run(
            ["ollama", "run", model],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        text = (out.stdout or "").strip()
        return text or f"(no output; stderr: {out.stderr.strip()[:200]})"
    except FileNotFoundError:
        sys.exit("ERROR: `ollama` not found on PATH.")
    except subprocess.TimeoutExpired:
        return f"(model {model} timed out after {timeout}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of .tf dev files")
    ap.add_argument("--n", type=int, default=3, help="how many files to test (default 3)")
    ap.add_argument(
        "--models", nargs=2,
        default=["glm-5.2:cloud", "deepseek-v4-pro:cloud"],
        help="the two explainer models to compare",
    )
    ap.add_argument("--save", default=None, help="optional path to also save the transcript")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".tf"))[: args.n]
    if not files:
        sys.exit(f"No .tf files found in {args.dir}")

    m1, m2 = args.models
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"INFORMAL EXPLAINER COMPARISON  --  {m1}  vs  {m2}")
    emit(f"{len(files)} file(s) from {args.dir}")
    emit("This is a human gut-check. Read both and judge faithfulness + usefulness.")
    emit("Finding text below uses the SAME scan() as your real pipeline.")
    emit("=" * 78)

    for i, fname in enumerate(files, 1):
        tf_path = os.path.join(args.dir, fname)
        with open(tf_path) as fh:
            tf = fh.read()
        nfail, finding, parse_ok = scan(tf_path)

        emit("\n" + "#" * 78)
        emit(f"# FILE {i}/{len(files)}: {fname}   ({nfail} failing check(s))")
        emit("#" * 78)
        emit("\n--- CHECKOV FINDING (what the raw output gives, i.e. Condition A) ---")
        emit(textwrap.indent(finding, "  "))

        if nfail <= 0:
            emit("\n(no parseable finding; skipping explainers for this file)")
            continue

        prompt = EXPLAIN_PROMPT.format(finding=finding, tf=tf)
        for m in (m1, m2):
            emit(f"\n--- EXPLANATION from {m} ---")
            emit(textwrap.indent(call_model(m, prompt), "  "))

        emit("\n>>> Your call: which explanation is more FAITHFUL (no invented claims)")
        emit(">>> and more USEFUL (would actually help nemo/an engineer fix it)?")

    emit("\n" + "=" * 78)
    emit("Done. Skim all files. If one model is consistently more grounded and")
    emit("actionable, that is your explainer. If it is a toss-up, pick GLM-5.2")
    emit("(stronger on independent benchmarks) and move on -- do not over-invest.")
    emit("=" * 78)

    if args.save:
        with open(args.save, "w") as fh:
            fh.write("\n".join(lines))
        print(f"\n(transcript saved to {args.save})")


if __name__ == "__main__":
    main()
