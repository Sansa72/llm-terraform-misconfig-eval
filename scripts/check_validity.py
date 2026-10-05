#!/usr/bin/env python3
"""
Lightweight validity guard for repaired Terraform files.

Checks three things, without needing Terraform or network access:
  1. The file still parses as valid HCL (not mangled/broken).
  2. The originally-flagged resource is still present (not deleted).
  3. Every resource reference (e.g. aws_kms_key.test.arn) points to a
     resource that is actually defined somewhere in the file.

"""

import sys
import re
import argparse
import hcl2


POLICY_VAR = re.compile(r'\$\{\s*(?:aws|s3|iam|sts|ec2|lambda|kms|dynamodb|sqs|sns|cognito-identity|saml)\s*:[^}]*\}')


def parse_file(path):
    """Try to parse the file as HCL. Returns (ok, data_or_error)."""
    try:
        with open(path) as fh:
            text = fh.read()
        text = POLICY_VAR.sub("POLICYVAR", text)
        data = hcl2.loads(text)
        return True, data
    except Exception as e:
        return False, str(e)


def _clean(key):
    """Some hcl2 parser versions leave literal quote characters in keys."""
    return key.strip().strip('"').strip("'")


def get_defined_resources(hcl_data):
    """
    Return a set of identifiers for everything defined in the file.
    Resources are stored as 'type.name'; data sources as 'data.type.name',
    matching how each is referenced in Terraform.
    """
    defined = set()
    for block in hcl_data.get("resource", []):
        for rtype, instances in block.items():
            rtype = _clean(rtype)
            for rname in instances.keys():
                defined.add(f"{rtype}.{_clean(rname)}")
    for block in hcl_data.get("data", []):
        for dtype, instances in block.items():
            dtype = _clean(dtype)
            for dname in instances.keys():
                defined.add(f"data.{dtype}.{_clean(dname)}")
    return defined


def _strip_noise(raw_text):
    """Remove comments and string/heredoc contents before scanning for
    references, so a resource-like token inside a comment or a string literal is
    not mistaken for a real dependency. An interpolation such as
    "${aws_kms_key.k.arn}" is preserved, because a genuine reference can live
    inside one. Without this step a model that merely mentions a resource in an
    explanatory comment, or embeds a resource-shaped substring in an ARN string,
    would have that phantom reference counted against it as a dangling one."""
    text = raw_text
    # Heredocs: <<EOF ... EOF (any marker), non-greedy to the matching terminator.
    text = re.sub(r'<<-?\s*(\w+)\b.*?^\s*\1\b', ' ', text, flags=re.DOTALL | re.MULTILINE)
    # Line comments (# and //) and block comments (/* */).
    text = re.sub(r'#.*', ' ', text)
    text = re.sub(r'//.*', ' ', text)
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
    # Double-quoted strings, keeping any ${...} interpolations they contain.
    def _keep_interp(m):
        return ' '.join(re.findall(r'\$\{[^}]*\}', m.group(0))) or ' '
    text = re.sub(r'"(?:[^"\\]|\\.)*"', _keep_interp, text)
    return text


def find_references(raw_text):
    """
    Find likely references in the file. Two forms:
      - resource refs:   aws_kms_key.test.arn        -> aws_kms_key.test
      - data refs:       data.aws_iam_policy_document.pass.json
                                                     -> data.aws_iam_policy_document.pass
    Comments and string literals are stripped first (see _strip_noise), so only
    references in real expression positions are considered. Heuristic scan rather
    than a full HCL expression evaluator, but reliable for the dangling-reference
    pattern this check exists to catch.
    """
    scan = _strip_noise(raw_text)
    found = set()

    # data.<type>.<name>.<attr>
    data_pat = re.compile(r'\bdata\.(aws_[a-z0-9_]+)\.([a-zA-Z0-9_-]+)\.[a-zA-Z0-9_]+\b')
    for m in data_pat.finditer(scan):
        found.add(f"data.{m.group(1)}.{m.group(2)}")

    # <type>.<name>.<attr> (resource refs) -- skip ones preceded by "data."
    res_pat = re.compile(r'(?<!data\.)\b(aws_[a-z0-9_]+)\.([a-zA-Z0-9_-]+)\.[a-zA-Z0-9_]+\b')
    for m in res_pat.finditer(scan):
        found.add(f"{m.group(1)}.{m.group(2)}")

    return found


def target_still_present(hcl_data, target):
    """
    Is the item Checkov flagged still in the file?

    target is 'type.name', optionally prefixed 'data.' for a data source.

    Checkov does not consistently mark data sources: it reports a data block as
    'aws_iam_policy_document.pass', with no 'data.' prefix, exactly as it reports
    a managed resource. An earlier version of this check took that at face value,
    looked only among the resource blocks, found nothing, and concluded the model
    had deleted the flagged item. Every file whose finding was raised against a
    data source was therefore marked invalid no matter what the model wrote. The
    check now looks in both places before declaring anything missing.
    """
    def _found_in(section, wanted_type, wanted_name):
        for block in hcl_data.get(section, []):
            for btype, instances in block.items():
                if _clean(btype) != wanted_type:
                    continue
                for bname in instances.keys():
                    if _clean(bname) == wanted_name:
                        return True
        return False

    if target.startswith("data."):
        _, dtype, dname = target.split(".", 2)
        return _found_in("data", dtype, dname)

    rtype, rname = target.split(".", 1)
    # Checkov omits the 'data.' prefix, so an unprefixed target may be either.
    return _found_in("resource", rtype, rname) or _found_in("data", rtype, rname)


def _dangling_in(path):
    """Return the set of references that are unresolved within a single file.
    Used to baseline: references already dangling in the original fixture are a
    property of the fixture, not of the repair, and must not count against it."""
    try:
        with open(path) as fh:
            raw = fh.read()
        ok, data = parse_file(path)
        if not ok:
            return set()
        defined = get_defined_resources(data)
        return {r for r in find_references(raw) if r not in defined}
    except Exception:
        return set()


def check_file(path, target_resource=None, original=None):
    result = {
        "file": path,
        "parses": False,
        "target_present": None,
        "dangling_references": [],
        "overall_valid": False,
        "notes": [],
    }

    with open(path) as fh:
        raw_text = fh.read()

    ok, data_or_err = parse_file(path)
    result["parses"] = ok
    if not ok:
        result["notes"].append(f"HCL parse failed: {data_or_err}")
        return result

    hcl_data = data_or_err
    defined = get_defined_resources(hcl_data)
    referenced = find_references(raw_text)

    preexisting = _dangling_in(original) if original else set()
    preexisting_types = {r.split(".", 1)[0] for r in preexisting}

    dangling = sorted(
        r for r in referenced
        if r not in defined
        and r not in preexisting
        and r.split(".", 1)[0] not in preexisting_types
    )
    result["dangling_references"] = dangling
    if dangling:
        result["notes"].append(
            f"Newly introduced references not defined in file: {', '.join(dangling)}"
        )

    if target_resource:
        present = target_still_present(hcl_data, target_resource)
        result["target_present"] = present
        if not present:
            result["notes"].append(
                f"Targeted item {target_resource} is missing (deleted or renamed)."
            )

    result["overall_valid"] = (
        result["parses"]
        and (result["target_present"] is not False)
        and not dangling
    )

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument(
        "--target-resource",
        help="type.name of the resource that was originally flagged, e.g. aws_efs_file_system.fail",
    )
    ap.add_argument(
        "--original",
        help="path to the original (pre-repair) file, so references already "
             "dangling in the fixture are not counted against the repair",
    )
    args = ap.parse_args()

    result = check_file(args.file, args.target_resource, args.original)

    print(f"File:              {result['file']}")
    print(f"Parses as HCL:     {result['parses']}")
    if args.target_resource:
        print(f"Target present:    {result['target_present']}")
    print(f"Dangling refs:     {result['dangling_references'] or 'none'}")
    print(f"OVERALL VALID:     {result['overall_valid']}")
    if result["notes"]:
        print("Notes:")
        for n in result["notes"]:
            print(f"  - {n}")

    sys.exit(0 if result["overall_valid"] else 1)


if __name__ == "__main__":
    main()