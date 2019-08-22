#%%
import re
import json
from io import StringIO
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


#%%
def read_index(index_file):
  with open(index_file) as f:
    patt = re.compile(r'^(?P<pdb_id>\w{4}).{15}([0-9.]{4}).*', re.MULTILINE)
    ids_pKs = np.array(re.findall(patt, f.read()))
  id2pK = {i: float(j) for i, j in ids_pKs}
  return id2pK


def RobustSanitizeMol(mol):
  """delete bad atom from mol
  """
  num = mol.GetNumAtoms()
  Chem.WrapLogs()
  stderr = sys.stderr
  for i in range(10):
    err = sys.stderr = StringIO()
    check = Chem.SanitizeMol(mol, catchErrors=True)
    if check == 0:
      break
    idx = re.findall("# (\d+)", err.getvalue())
    idx = int(idx[0])
    bad = mol.GetAtomWithIdx(idx)
    bad.SetAtomicNum(0)
    for n in bad.GetNeighbors():
      if n.GetAtomicNum() == 1:
        n.SetAtomicNum(0)
    mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#0]'))
  sys.stderr = stderr
  num = num - mol.GetNumAtoms()
  if check == 0:
    # print("Done. delete {0} atom in RobustSanitizeMol().".format(num))
    return mol
  else:
    print("Fail.")
    return None


def getProp(mol):
  mw = Descriptors.ExactMolWt(mol)
  logp = Descriptors.MolLogP(mol)
  rotb = Descriptors.NumRotatableBonds(mol)
  hbd = Descriptors.NumHDonors(mol)
  hba = Descriptors.NumHAcceptors(mol)
  q = Chem.GetFormalCharge(mol)
  return tuple([mw, logp, rotb, hbd, hba, q])


#%%
# index_file = 'pocket_clust/index/INDEX_core_data.2013'
index_file = 'pocket_clust/index/INDEX_refined_data.2015'
id2pK = read_index(index_file)

pdbbind = Path("/home/yangjc/tmp/pdbbind/v2015")
X = []
y = []
props = []
atomic_nums = []
for id_, pK in id2pK.items():
  pocket = pdbbind / id_ / (id_ + '_pocket.pdb')
  mol = Chem.MolFromPDBFile(str(pocket), removeHs=False, sanitize=False)
  mol = RobustSanitizeMol(mol)
  if mol is None:
    continue
  p = getProp(mol)
  props.append(p)
  atomic_nums.append(Counter([a.GetAtomicNum() for a in mol.GetAtoms()]))
  y.append(pK)

atom_types = sorted(sum(atomic_nums, Counter()).keys())
print(atom_types)
# num_features = len(props[])
for p, at_nums in zip(props, atomic_nums):
  at_count = tuple([at_nums[t] if t in at_nums else 0 for t in atom_types])
  # print(at_count)
  X.append(p + at_count)
X = np.array(X)
y = np.array(y)

#%%
for seed in (111, 222, 333):
  np.random.seed(seed)
  N = len(X)
  perm = np.random.permutation(N)
  train_idx = perm[:int(N * 0.9)]
  # valid_idx = perm[int(N * 0.8):int(N * 0.9)]
  test_idx = perm[int(N * 0.9):]
  train_X = X[train_idx]
  test_X = X[test_idx]
  train_y = y[train_idx]
  test_y = y[test_idx]

  clf = RandomForestRegressor(
      n_estimators=20,
      max_depth=15,
      # min_samples_split=10,
      min_samples_split=5,
      min_samples_leaf=10,
      random_state=0,
      n_jobs=8,
  )
  # from sklearn.neural_network import MLPRegressor
  # clf = MLPRegressor(
  #     hidden_layer_sizes=(100, 50, 10),
  #     early_stopping=True,
  #     validation_fraction=0.1,
  # )
  clf.fit(train_X, train_y)
  train_r2 = clf.score(train_X, train_y)
  test_r2 = clf.score(test_X, test_y)
  print('seed {} r2: train {} test {}'.format(seed, train_r2, test_r2))

#%%
