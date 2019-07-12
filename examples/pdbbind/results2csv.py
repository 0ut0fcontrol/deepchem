"""convert results in json to table in csv
"""
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("-j", '--json', nargs='+',required=True)
parser.add_argument("-o", required=True)
args = parser.parse_args()

data = []
for j in args.json:
    with open(j) as f:
        d = json.load(f)
    row = [d[i] for i in ('version', 'subset', 'split', 'component', 'seed')]
    for _set, metrics in d['best_scores'].items():
        for m, v in metrics.items():
            data.append(row + [_set, m, v])
cols = ('version', 'subset', 'split', 'component', 'seed', 'set', 'metric', 'value')
df = pd.DataFrame(data, columns=cols)
df.to_csv(args.o, index=False)

