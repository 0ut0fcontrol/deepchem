"""computing protein sequence similarity for PDBbind using NWalign.
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
    '-e', '--exe', default='./NWalign', help='NWalign exe, default: ./NWalign')
args = parser.parse_args()

data_dir = Path(args.data_dir)

with open(args.index) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).*', re.MULTILINE)
  index = np.array(re.findall(patt, f.read()))
# use np.array and np.int32 for less memory and faster process fork in imap
pairs = np.array(np.triu_indices(len(index), 1), dtype=np.int32).T

aligned_ident_pattern = re.compile(
    r".*Aligned length:\s+(\d+)\nIdentical length:\s+(\d+).*")


def NWalign(pair):
  i, j = pair
  id_i, id_j = index[i], index[j]
  pdb_i = str(data_dir / id_i / (id_i + '_protein.pdb'))
  pdb_j = str(data_dir / id_j / (id_j + '_protein.pdb'))
  cmd = (args.exe, pdb_i, pdb_j, '1')
  try:
    job = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = job.stdout.decode('utf-8')
    matchs = re.findall(aligned_ident_pattern, log)
    aligned_len, ident_len = matchs[0]
    identity = float(ident_len) / float(aligned_len)
    return identity
  except:
    # print(cmd)
    return 0.


# use np.float16 for less memory
simi_triu = np.zeros(len(pairs), dtype=np.float16)
p = Pool()
jobs = tqdm(
    p.imap(NWalign, pairs, chunksize=100),
    desc='NWalign',
    total=len(pairs),
    smoothing=0,
)

for idx, identity in enumerate(jobs):
  simi_triu[idx] = identity
p.close()

dist_triu = 1 - simi_triu

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Z = linkage(dist_triu)
fig = plt.figure(figsize=(25, 10))
if len(index) < 1000:
  dn = dendrogram(Z)
  plt.savefig(args.output + '.png')

with open(args.output + f".simi.json", 'w') as f:
  result = {
      'index': index.tolist(),
      'simi_triu': simi_triu.tolist(),
  }
  json.dump(result, f)

for simi_cutoff in (0.5, 0.6, 0.7, 0.8, 0.9):
  dist_cutoff = 1 - simi_cutoff
  cluster_mask = fcluster(Z, t=dist_cutoff, criterion='distance')
  cluster_nums = set(cluster_mask)
  clusters = []
  for num in cluster_nums:
    clust = list(index[cluster_mask == num])
    clusters.append(clust)
  json_name = args.output + f'.clust.ident{simi_cutoff:.1f}.json'
  with open(json_name, 'w') as f:
    result = {
        'clusters': clusters,
    }
    json.dump(result, f, indent=4)
