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
parser.add_argument(
    '-d', '--data_dir', required=True, help='PDBbind data dir, like ./v2015')
parser.add_argument(
    '-c', type=float, default=0.8, help='TMscore cutoff for cluster')
parser.add_argument(
    '-e', '--exe', default='./TMalign', help='TMalign exe, default: ./TMalign')
args = parser.parse_args()

with open(args.index) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).*', re.MULTILINE)
  index = re.findall(patt, f.read())
index = index
idx = np.arange(len(index)).reshape(-1,1)
data_dir = Path(args.data_dir)

def TMdist(i,j):
  [i], [j] = i, j
  id_i, id_j = index[int(i)], index[int(j)]
  pdb_i = str(data_dir / id_i / (id_i + '_pocket.pdb'))
  pdb_j = str(data_dir / id_j / (id_j + '_pocket.pdb'))
  cmd = (args.exe, pdb_i, pdb_j, '-a')
  try:
    job = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = job.stdout.decode('utf-8')
    # TM-score= 0.82668 (if normalized by average length of two structures, i.e., LN= 48.00, d0= 2.18)
    patt = re.compile(r'TM-score= ([0-9.]+) \(if normalized by average.*')
    matchs = re.findall(patt, log)
    TMscore = float(matchs[0])
    return 1 - TMscore
  except:
    print(cmd)
    return 1.
from sklearn.neighbors import NearestNeighbors
start = dt.now()
tree = NearestNeighbors(n_neighbors=3, radius=0.8, metric=TMdist, leaf_size=3, n_jobs=-1)
tree.fit(idx)
print('time to build tree: {}'.format(dt.now() - start))
neigb = tree.radius_neighbors(idx)
raise SystemExit()

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Z = linkage(idx, metric=TMdist)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.savefig(args.output + '.png')

dist_cutoff = 1 - args.c
cluster_mask = fcluster(Z, t=dist_cutoff, criterion='distance')
index = np.array(index)
cluster_ids = []
for i in range(1, max(cluster_mask) + 1):
  ids = list(index[cluster_mask == i])
  cluster_ids.append(ids)

with open(args.output, 'w') as f:
  json.dump(cluster_ids, f)
