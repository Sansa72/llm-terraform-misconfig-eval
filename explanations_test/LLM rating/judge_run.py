#!/usr/bin/env python3
"""
judge_run.py

LLM-judge pass for the agreement study. Scores the SAME 45 items the human
raters see (read directly from sampled_bundle.txt, so the content is
byte-identical) against the SAME five rubric statements used in the form.
The 45 include the five catch items at their blinded positions; the judge,
like the human raters, is not told which items are catches.

Output: judge_scores.csv with columns rater,item_id,dimension,score,notes
(rater is always "JUDGE"), matching the human CSVs for the kappa analysis.

Usage:
    python3 judge_run.py --bundle sampled_bundle.txt --out judge_scores.csv

Safe to re-run: already-scored items in the output file are skipped, so an
interrupted run resumes where it stopped.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request

# ----------------------------------------------------------------------
# CONFIG — check these two before running
# ----------------------------------------------------------------------
OLLAMA_URL  = "http://localhost:11434/api/chat"
JUDGE_MODEL = "glm-5.2:cloud"     # confirm the exact tag with `ollama list`
TEMPERATURE = 0.0
EXPECTED_ITEMS = 45              # the 45-item bundle: 40 rated + 5 catch
MAX_RETRIES = 3
RETRY_WAIT_S = 10

# Statements: IDENTICAL wording to the Google Form / reading packet.
DIMS = [
    ("technical_soundness",
     "The explanation is technically correct about the misconfiguration and its consequences."),
    ("evidence_grounding",
     "Every claim is supported by the Terraform file shown; nothing is asserted that the evidence does not support."),
    ("clarity",
     "The explanation is clear and understandable to an engineer who has to act on it."),
    ("remediation_safety",
     "The remediation described is safe to apply and would not break the resource or introduce new risk."),
    ("overall_quality",
     "Overall, this is a good explanation of the finding."),
]

SYSTEM = (
    "You are an experienced cloud security engineer acting as a rater in an "
    "evaluation study. You will be shown a Terraform file and an explanation of "
    "the security problems a scanner found in it. Rate the explanation against "
    "five statements on a five-point Likert scale: "
    "1 = strongly disagree, 2 = disagree, 3 = neither agree nor disagree, "
    "4 = agree, 5 = strongly agree. "
    "Judge strictly from the material provided. Check the explanation's claims "
    "against the Terraform file; do not reward fluent writing the evidence does "
    "not support. Respond only with the JSON object requested."
)

def user_prompt(tf, expl):
    dims = "\n".join(f'- "{k}": {t}' for k, t in DIMS)
    return f"""Rate the EXPLANATION below against each statement, on the 1-5 scale.

Statements:
{dims}

Respond with a single JSON object mapping each statement key to an integer 1-5,
plus a "notes" field of at most two sentences justifying your lowest score.
No other text, no markdown fences.

--- TERRAFORM FILE ---
{tf}

--- EXPLANATION ---
{expl}
"""

def parse_bundle(path):
    txt = open(path, encoding="utf-8").read()
    parts = re.split(r'^===== (item_\d+) =====$', txt, flags=re.M)
    items = []
    for i in range(1, len(parts), 2):
        iid, body = parts[i], parts[i + 1]
        tf = body.split("----- ORIGINAL TERRAFORM FILE -----")[1] \
                 .split("----- P2 EXPLANATION -----")[0].strip()
        ex = body.split("----- P2 EXPLANATION -----")[1].strip()
        items.append((iid, tf, ex))
    return items

def call_judge(prompt):
    payload = {
        "model": JUDGE_MODEL,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    body = json.dumps(payload).encode()
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
            text = data["message"]["content"].strip()
            # Reasoning models may emit a monologue before the JSON, and that
            # monologue can itself contain braces. Take the LAST balanced
            # top-level {...} object rather than first-{ to last-}.
            end = text.rfind("}")
            if end == -1:
                raise ValueError("no JSON object in response")
            depth, start = 0, -1
            for i in range(end, -1, -1):
                if text[i] == "}":
                    depth += 1
                elif text[i] == "{":
                    depth -= 1
                    if depth == 0:
                        start = i
                        break
            if start == -1:
                raise ValueError("unbalanced JSON in response")
            obj = json.loads(text[start:end + 1])
            # validate
            for k, _ in DIMS:
                v = obj.get(k)
                if not isinstance(v, int) or not 1 <= v <= 5:
                    raise ValueError(f"bad or missing score for {k}: {v!r}")
            return obj
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(RETRY_WAIT_S)
    raise RuntimeError(f"judge failed after {MAX_RETRIES} attempts: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="sampled_bundle.txt")
    ap.add_argument("--out", default="judge_scores.csv")
    args = ap.parse_args()

    items = parse_bundle(args.bundle)
    print(f"bundle: {len(items)} items")
    if len(items) != EXPECTED_ITEMS:
        sys.exit(f"ERROR: expected {EXPECTED_ITEMS} items but bundle has "
                 f"{len(items)} - wrong or stale bundle file? "
                 f"({os.path.abspath(args.bundle)})")

    # resume support: skip items already in the output
    done = set()
    if os.path.exists(args.out):
        counts = {}
        with open(args.out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[row["item_id"]] = counts.get(row["item_id"], 0) + 1
        done = {iid for iid, c in counts.items() if c == len(DIMS)}
        partial = {iid for iid, c in counts.items() if c != len(DIMS)}
        if partial:
            sys.exit(f"ERROR: partially scored items in {args.out}: "
                     f"{sorted(partial)} - remove their rows before resuming, "
                     f"so no item is half-scored.")
        print(f"resuming: {len(done)} items already scored")

    write_header = not os.path.exists(args.out)
    out = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(out, fieldnames=["rater", "item_id", "dimension", "score", "notes"])
    if write_header:
        w.writeheader()

    for idx, (iid, tf, ex) in enumerate(items, 1):
        if iid in done:
            continue
        print(f"[{idx}/{len(items)}] {iid}")
        obj = call_judge(user_prompt(tf, ex))
        notes = str(obj.get("notes", ""))[:500]
        for k, _ in DIMS:
            w.writerow({"rater": "JUDGE", "item_id": iid,
                        "dimension": k, "score": obj[k], "notes": notes})
        out.flush()

    out.close()
    print(f"done -> {args.out}")

if __name__ == "__main__":
    main()