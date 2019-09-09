"""clustering pdbbind based on ligand fingerprints.
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

from rdkit import Chem
from rdkit.Chem import AllChem

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('index', help='PDBbind data index')
parser.add_argument('output', help='output data in json')
parser.add_argument(
    '-d', '--data_dir', required=True, help='PDBbind data dir, like ./v2015')
args = parser.parse_args()
start = dt.now()


def ClusterFps(fps, cutoff=0.2):
  # (ytz): this is directly copypasta'd from Greg Landrum's clustering example.
  dists = []
  nfps = len(fps)
  from rdkit import DataStructs
  for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])
  from rdkit.ML.Cluster import Butina
  cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
  return cs


data_dir = Path(args.data_dir)

with open(args.index) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).*', re.MULTILINE)
  codes = np.array(re.findall(patt, f.read()))
# use np.array and np.int32 for less memory and faster process fork in imap

fail_codes = []
succeeded_codes = []
fps = []
for code in codes:
  # rdkit read mol2 fail a lot.
  # Morgan Fingerprint need sanitize=True, so use ligand.pdb coverted by "babel" from ligand.mol2
  # http://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
  pdb = str(data_dir / code / (code + '_ligand.pdb'))
  mol = Chem.MolFromPDBFile(str(pdb))
  if mol is None:
    print(
        f"WARNING: RDKit failed to load ligand of {code}, it will be assigned to validation set."
    )
    fail_codes.append(code)
  else:
    succeeded_codes.append(code)
    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
print(f'Fail to load {len(fail_codes)}/{len(codes)} ligands.\n')
succeeded_codes = np.array(succeeded_codes)

for simi_cutoff in (0.5, 0.6, 0.7, 0.8, 0.9):
  dist_cutoff = 1 - simi_cutoff
  print(f"Start cluster ligand with cutoff {simi_cutoff}")
  clust_indices = ClusterFps(fps, cutoff=dist_cutoff)
  clusters = [succeeded_codes[list(inds)].tolist() for inds in clust_indices]
  json_name = f'{args.output}.simi{simi_cutoff:.1f}.clust.json'
  with open(json_name, 'w') as f:
    json.dump(clusters, f, indent=4)
  print(f"Result saved to {json_name}\n")

print(f'Elapsed time {dt.now() - start}.')