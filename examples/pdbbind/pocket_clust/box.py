#%%
import re
import json
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd

#%%
tips = sns.load_dataset("tips")
ax = sns.boxplot(x="day", y="total_bill", data=tips)
ax = sns.swarmplot(
    x="day",
    y="total_bill",
    data=tips,
    color=".25",
)

#%%
print(Path.cwd())
index_file = 'pocket_clust/index/INDEX_core_data.2013'
clust_file = 'pocket_clust/cluster_ids/INDEX_core_data.2013.TMscore0.5.json'
with open(index_file) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).{15}([0-9.]{4}).*', re.MULTILINE)
  index = np.array(re.findall(patt, f.read()))
print(index[:3])
index2pk = {i: float(j) for i, j in index}
#%%
with open(clust_file) as f:
  clust_ids = json.load(f)
# print(clust_ids)

#%%
data = []
# N = len(index)
# test_cut =
for i, clust in enumerate(sorted(clust_ids, key=len)):
  # print(clust)
  mean_pK = np.mean([index2pk[id_] for id_ in clust])
  for id_ in clust:
    data.append((i, mean_pK, id_, len(clust), index2pk[id_]))

df = pd.DataFrame(columns=('clust_num', 'mean_pK', 'id', 'size', 'pK'), data=data)

#%%
# ax = sns.boxplot(x="mean_pK", y="pK", data=df)
ax = sns.swarmplot(
    x="mean_pK",
    y="pK",
    data=df,
    color=".25",
)

#%%
index_file = 'pocket_clust/index/INDEX_refined_data.2015'
clust_file = 'pocket_clust/cluster_ids/INDEX_refined_data.2015.TMscore0.5.json'
with open(index_file) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).{14}([ 0-9.]{5}).*', re.MULTILINE)
  index = np.array(re.findall(patt, f.read()))
print(index[:3])
index2pk = {i: float(j) for i, j in index}
#%%
with open(clust_file) as f:
  clust_ids = json.load(f)
# print(clust_ids)

#%%
data = []
# N = len(index)
small_clust = []
for i, clust in enumerate(sorted(clust_ids, key=len)):
  # print(clust)
  mean_pK = np.mean([index2pk[id_] for id_ in clust])
  if len(clust) < 30:
      # small_clust.append(clust)
      i = 0
  for id_ in clust:
    data.append((i, mean_pK, id_, len(clust), index2pk[id_]))
df = pd.DataFrame(columns=('clust_num', 'mean_pK', 'id', 'size', 'pK'), data=data)

#%%
ax = sns.boxplot(x="clust_num", y="pK", data=df)
ax = sns.swarmplot(
    x="clust_num",
    y="pK",
    data=df,
    color=".25",
)


#%%
index_file = 'pocket_clust/index/INDEX_refined_data.2015'
clust_file = '/home/yangjc/git/can-ai-do/pdbbind/usearch_result/INDEX_refined_data.2015.uc'
with open(index_file) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).{14}([ 0-9.]{5}).*', re.MULTILINE)
  index = np.array(re.findall(patt, f.read()))
print(index[:3])
index2pk = {i: float(j) for i, j in index}
#%%
with open(clust_file) as f:
  clust_ids = {}
  for line in f:
      if line[0] != 'C':
          fields = line.split()
          clust_num = int(fields[1])
          id_ = fields[8]
          if clust_num in clust_ids:
            clust_ids[clust_num].append(id_)
          else:
            clust_ids[clust_num] = [id_]
print(clust_ids)

#%%
data = []
# N = len(index)
small_clust = []
for i in sorted(clust_ids.keys()):
  clust = clust_ids[i]
  # print(clust)
  mean_pK = int(np.mean([index2pk[id_] for id_ in clust]) * 10)
  if len(clust) < 30:
      # small_clust.append(clust)
      i = 0
      mean_pK = 0
  for id_ in clust:
    data.append((i, mean_pK, id_, len(clust), index2pk[id_]))
df = pd.DataFrame(columns=('clust_num', 'mean_pK', 'id', 'size', 'pK'), data=data)

#%%
ax = sns.boxplot(x="clust_num", y="pK", data=df)
ax = sns.swarmplot(
    x="clust_num",
    y="pK",
    data=df,
    color=".25",
)

#%%
#ax = sns.boxplot(x="mean_pK", y="pK", data=df)
ax = sns.swarmplot(
    x="mean_pK",
    y="pK",
    data=df,
    color=".25",
)

#%%
clust_selected = df.loc[df['clust_num']==131]
clust_selected.sort_values(by='pK')