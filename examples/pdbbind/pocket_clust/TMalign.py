"""computing TMalign pocket similarity for PDBbind.
"""
import re
import json
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime as dt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('index', help='PDBbind data index')
parser.add_argument('output', help='output data in json')
parser.add_argument('-d',
                    '--data_dir',
                    required=True,
                    help='PDBbind data dir, like ./v2015')
parser.add_argument('-e',
                    '--exe',
                    default='./TMalign',
                    help='TMalign exe, default: ./TMalign')
args = parser.parse_args()

with open(args.index) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).*', re.MULTILINE)
  index = np.array(re.findall(patt, f.read()))
data_dir = Path(args.data_dir)

# use np.array and np.int32 for less memory and faster process fork in imap
cmds = np.array(np.triu_indices(len(index), 1), dtype=np.int32).T

TMscore_patt = re.compile(r'TM-score= ([0-9.]+) \(if normalized by average.*')


def TMalign(cmd):
  i, j = cmd
  id_i, id_j = index[i], index[j]
  pdb_i = str(data_dir / id_i / (id_i + '_pocket.pdb'))
  pdb_j = str(data_dir / id_j / (id_j + '_pocket.pdb'))
  cmd = (args.exe, pdb_i, pdb_j, '-a')
  try:
    job = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = job.stdout.decode('utf-8')
    # TM-score= 0.82668 (if normalized by average length of two structures, i.e., LN= 48.00, d0= 2.18)
    matchs = re.findall(TMscore_patt, log)
    TMscore = float(matchs[0])
    return 1. - TMscore
  except:
    # print(cmd)
    return 1.


# use np.float16 for less memory
dist_triu = np.zeros(len(cmds), dtype=np.float16)
p = Pool()
jobs = tqdm(p.imap(TMalign, cmds, chunksize=100),
            desc='TMalign',
            total=len(cmds),
            smoothing=0)

for idx, TMscore in enumerate(jobs):
  dist_triu[idx] = TMscore
p.close()

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Z = linkage(dist_triu)
fig = plt.figure(figsize=(25, 10))
try:
  dn = dendrogram(Z)
  plt.savefig(args.output + '.png')
except:
  print('Fail on generate dendrogram.')

# TM dist 0.2, TM score 0.8 can split most of clusteres in core set
# TM dist 0.5, TM score 0.5 mean same fold.
for TMscore_cutoff in (0.5, 0.8):
  TMdist_cutoff = 1 - TMscore_cutoff
  cluster_mask = fcluster(Z, t=TMdist_cutoff, criterion='distance')
  cluster_ids = []
  for i in range(1, max(cluster_mask) + 1):
    ids = list(index[cluster_mask == i])
    cluster_ids.append(ids)
  json_name = args.output + f'.TMscore{TMscore_cutoff:.1f}' + '.json'
  with open(json_name, 'w') as f:
    json.dump(cluster_ids, f, indent=4)
