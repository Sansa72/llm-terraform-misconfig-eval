import os, re, csv
# Run from the Thesis root: python3 scripts/label_families.py
from collections import Counter

FAM = [('encryption', r'encrypt|kms|cmk|key|tls|ssl'),
       ('iam',        r'iam|policy|admin|privilege|wildcard|principal'),
       ('logging',    r'log|monitor|trail|audit|alarm'),
       ('backup',     r'backup|snapshot|retention|version'),
       ('network',    r'public|ingress|egress|sg_|security_group|acl|vpc')]

def fam(fn):
    n = fn.lower()
    for name, pat in FAM:
        if re.search(pat, n):
            return name
    return 'residual'

pops = {}
pops['source'] = [f for f in os.listdir('dataset/raw')
                  if f.endswith('.tf') and 'checks_' in f]
pops['misconf'] = [os.path.basename(r[0]) for r in
                   csv.reader(open('dataset/labels.csv')) if r]
for k, d in [('test', 'dataset/test'), ('dev', 'dataset/dev'),
             ('held', 'dataset/surplus')]:
    pops[k] = [f for f in os.listdir(d) if f.endswith('.tf')]

order = ['encryption', 'iam', 'logging', 'network', 'backup', 'residual']
counts = {}
for k in pops:
    counts[k] = Counter(fam(f) for f in pops[k])

print('totals:', dict((k, len(v)) for k, v in pops.items()))
header = '%-12s' % 'family'
for k in pops:
    header += '%14s' % k
print(header)
for fm in order:
    row = '%-12s' % fm
    for k in pops:
        n = counts[k][fm]
        pct = 100.0 * n / len(pops[k])
        row += '%8d (%3.0f%%)' % (n, pct)
    print(row)
