# Do LLM Explanations Help AI Models Fix Insecure Terraform?

Code and results from my MSc Information Security dissertation at UCL, *Evaluating LLM Explanations for Terraform Misconfiguration Findings* (supervised by Earl Barr, 2026).

## The question

Security scanners like [Checkov](https://github.com/bridgecrewio/checkov) flag misconfigured infrastructure-as-code, but their findings are terse. This project tests whether an LLM-written explanation of a finding helps an AI "Fixer" model repair the Terraform file better than the scanner's finding alone.

## Headline results

On a 100-file test set, each file run three times with `codegemma:7b` as the Fixer:

| Condition | What the Fixer sees | Valid repair rate |
|---|---|---|
| **C** (floor) | The Terraform file only | 19.7% |
| **A** (baseline) | File + raw Checkov finding | 48.0% |
| **B** (test) | File + LLM explanation of the finding | **56.7%** |

- **B vs A: +8.7 percentage points** (95% CI [2.0, 15.3], p = 0.011).
- **Gaming nearly trebled** under explanations (pooled rate 4.0% → 11.3%): the Fixer more often produced files that passed the scanner without being valid Terraform.
- **No benefit for stronger Fixers.** With `nemotron-3-super` the B − A gap was +0.3 pp; with `kimi-k2.7-code` it was −3.7 pp.

**Takeaway:** explanations help a weak model, but they also make it better at satisfying the scanner without genuinely fixing the file.

## How it works

```
Misconfigured .tf file
        │
        ▼
   Checkov scan ──► finding
        │
        ▼
 Explainer LLM (deepseek-v4-pro) ──► explanation        [generate_explanations.py]
        │
        ▼
 Fixer LLM, up to 5 attempts, under condition C / A / B  [fullfix_conditions.py]
        │
        ▼
 Re-scan with Checkov + validity check                   [check_validity.py]
        │
        ▼
 fixed / gamed_scanner / not_fixed  ──► CSV ──► statistics scripts
```

**Outcome labels in every results CSV:**

| Outcome | Meaning |
|---|---|
| `fixed` | Checkov reports zero failing checks **and** the file is still valid Terraform |
| `gamed_scanner` | Checkov reports zero failures, but the file is broken, so Checkov silently skipped it |
| `not_fixed` | Checks still fail after 5 attempts |

---

## Setup

### 1. Install Python packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install checkov python-hcl2 pandas numpy scipy
```

> ⚠️ **`python-hcl2` is required.** `fullfix_conditions.py` calls `check_validity.py` to confirm each repair is valid Terraform. If `python-hcl2` is missing, the validity check silently passes every file, so broken repairs get counted as `fixed`.
>
> Check it works before running anything:
> ```bash
> python3 scripts/check_validity.py dataset/test/<any_file>.tf
> ```
> You should see `OVERALL VALID: True`, not an import error.

<!-- TODO: pin the Checkov version used in the thesis, e.g. pip install checkov==X.Y.Z -->

### 2. Set up Ollama

All models run through [Ollama](https://ollama.com/). The scripts call its local API at `http://localhost:11434`.

```bash
ollama serve                 # leave running in a separate terminal
ollama pull codegemma:7b     # the main Fixer
```

The Explainer (`deepseek-v4-pro:cloud`), the Judge (`glm-5.2:cloud`) and the stronger Fixers are Ollama cloud models. These need you to be signed in to an Ollama account.

**Run all commands from the repository root.**

---

## Reproducing the experiment

### Step 1 — Generate explanations

```bash
python3 scripts/generate_explanations.py \
    --model deepseek-v4-pro:cloud \
    --dir dataset/test \
    --outdir explanations_test \
    --prompts p7_grounded
```

- Writes one `<file>.tf.txt` explanation per Terraform file into `explanations_test/p7_grounded/`.
- Also writes `manifest.csv`, which records whether each explanation leaked Terraform code (`has_hcl`).
- `p7_grounded` is the prompt used for the main results. Leave out `--prompts` to run all 12 prompts.
- Add `--limit 3` for a quick test run.

<details>
<summary>All prompt IDs</summary>

| ID | Strategy |
|---|---|
| `p0_neutral` | Neutral prompt, used only to compare explainer models |
| `p1_minimal` | Bare instruction, the floor |
| `p2_cot` | Chain-of-thought |
| `p3_playbook` | Step-by-step remediation briefing |
| `p4_plansolve` | Plan-and-solve |
| `p5_stepback` | Step-back: general principle first |
| `p6_contrastive` | Contrast broken vs secure configuration |
| `p7_grounded` | **Every claim tied to the scanner evidence (main prompt)** |
| `p8_ruleref` | Adds Checkov's rule references |
| `p9_verify` | Chain-of-verification |
| `p10_constitutional` | Self-critique against a rubric |
| `p11_schema` | Structured JSON output |
| `p12_combined` | Grounded + playbook combined |

</details>

### Step 2 — Run the Fixer under each condition

Run each condition three times, changing the run number in `--out`:

```bash
# Condition C: file only
python3 scripts/fullfix_conditions.py --model codegemma:7b \
    --dir dataset/test --condition C --out test_C_run1.csv

# Condition A: file + Checkov finding
python3 scripts/fullfix_conditions.py --model codegemma:7b \
    --dir dataset/test --condition A --out test_A_run1.csv

# Condition B: file + explanation
python3 scripts/fullfix_conditions.py --model codegemma:7b \
    --dir dataset/test --condition B \
    --explanations explanations_test/p7_grounded \
    --out test_B_p7_run1.csv
```

- Each file gets up to **5 attempts**. After a failed attempt, A and B are told which checks still fail; C is only told "try again".
- The script prints a running tally and writes one row per file to the CSV.
- Each attempt's output is kept in a temporary folder, printed at the start of the run.
- For slow cloud Fixers, raise `OLLAMA_TIMEOUT` at the top of the script.

### Step 3 — Put the CSVs where the statistics scripts expect them

| Runs | Folder | File names |
|---|---|---|
| codegemma, A / B / C | `real_test/p7 test/` | `test_A_run1.csv` … `test_C_run3.csv`, `test_B_p7_run1.csv` … |
| codegemma, prompt p12 | `real_test/p12 test/` | `test_B_p12_run1.csv` … |
| codegemma, prompt p3 | `data/` | `test_B_p3_run1.csv` … |
| nemotron | `real_test/p7 nemotron test/` | `gen_nemotron_A_run1.csv` … |
| kimi | `real_test/p7 kimi test/` | `gen_kimi_A_run1.csv` … |

The results from the thesis are already in these folders, so you can skip Steps 1–2 and go straight to Step 4.

### Step 4 — Reproduce the statistics

Each script prints its result next to the number reported in the thesis, so you can check they match.

| Script | Reproduces |
|---|---|
| `condition_ci.py` | Repair rate per condition (C 19.7%, A 48.0%, B 56.7%) with 95% CIs |
| `ci_check.py` | Paired B − A (+8.7 pp) and A − C differences |
| `gamed_ci.py` | Gaming rate per condition and paired differences |
| `gamed_ci_fixers.py` | Gaming rates for all three Fixers |
| `extra_ci.py` | Prompt comparison (p3, p7, p12) and the stronger-Fixer robustness check |
| `family_ci.py` | B − A broken down by misconfiguration family |
| `p12_ci.py` | p12 vs p7 |
| `sample_size_calc.py` | Sample size for the rating study (32 / 36 / 40 items) |

```bash
python3 scripts/condition_ci.py
```

`p12_vs_p3_ci.py` takes the six CSV paths as arguments, the three p12 runs then the three p3 runs:

```bash
python3 scripts/p12_vs_p3_ci.py \
    "real_test/p12 test/test_B_p12_run1.csv" "real_test/p12 test/test_B_p12_run2.csv" "real_test/p12 test/test_B_p12_run3.csv" \
    data/test_B_p3_run1.csv data/test_B_p3_run2.csv data/test_B_p3_run3.csv
```

**Not runnable from this repo:** `kappa_check.py` and `rho_check.py` compute rater agreement and need the human ratings, which are not published (see Notes).

---

## Other tools

**Check a single repaired file:**

```bash
python3 scripts/check_validity.py repaired.tf \
    --target-resource aws_s3_bucket.example \
    --original dataset/test/original.tf
```

Checks that the file parses, the flagged resource wasn't deleted, and no references point to resources that don't exist. Exits with `0` if valid, `1` if not.

**Count misconfiguration families in the dataset:**

```bash
python3 scripts/label_families.py
```

**Compare two explainer models side by side** (informal, for reading by eye):

```bash
python3 scripts/compare_explainers.py --dir dataset/dev --n 3 --save comparison.txt
```

---

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/` | All experiment and analysis scripts |
| `dataset/` | Terraform files: `dev/` (30), `test/` (100), plus source files and labels |
| `explanations/`, `explanations_test/` | Generated explanations for the dev and test sets |
| `real_test/` | Fixer results on the test set (the CSVs the thesis numbers come from) |
| `data/` | Prompt p3 results and item mappings |
| `AC testing/`, `B testing/` | Development-set runs for each condition |
| `local models/`, `cloud models/` | Fixer candidate screening results |
| `expl_cmp_deepseek/`, `expl_cmp_glm/` | Explainer model comparison |

## Notes

- **Human ratings are not included**, to protect the raters' privacy. Agreement statistics are reported in the dissertation.
- Terraform fixtures are derived from the Checkov project (Apache-2.0). <!-- TODO: confirm source -->

## Tech used

Python · Checkov · Terraform · Ollama · LLM-as-a-judge · statistics (paired t-tests, bootstrap CIs, Cohen's and Fleiss' kappa)

## Author

Iacopo Boaron Otero · <!-- TODO: GitHub / LinkedIn / Medium links -->
