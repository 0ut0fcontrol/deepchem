"""
Script that trains Atomic Conv models on PDBbind dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import json
import shutil
import argparse
import random
import numpy as np
import tensorflow as tf
import deepchem as dc
from datetime import datetime as dt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-max_epoch", type=int, default=100)
parser.add_argument("-patience", type=int, default=3)
parser.add_argument("-version", default='2015')
parser.add_argument("-subset", default='core')
parser.add_argument(
    "-test_core",
    help=("use core set as test set, 2015 or 2018"
          "split test set from subset if not given."))
parser.add_argument("-component", default='binding')
parser.add_argument("-split", default=None)
parser.add_argument("-seed", type=int, default=111)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-clust_file")
parser.add_argument("-save_dir", default='/tmp')
parser.add_argument("-data_dir")
parser.add_argument("-reload", action='store_true')
parser.add_argument(
    "-shuffle", action='store_true', help='Shuffling or Curriculum Learning')
parser.add_argument("-shuf_labels", action='store_true')
parser.add_argument("-trans", action='store_true')
parser.add_argument("-feat_only", action='store_true')
parser.add_argument("-same_protein", action='store_true')
parser.add_argument("-same_ligand", action='store_true')
parser.add_argument("-timestamp", action='store_true')
parser.add_argument("-split_complex", action='store_true')
args = parser.parse_args()

start = dt.now()

# np seed for split only
np.random.seed(args.seed)
# tf seed not work, every training will different.
tf.set_random_seed(args.seed)

frag1_num_atoms = 368  # for ligand atoms with Hs.
frag2_num_atoms = 1350  # for pocket atoms with Hs
# frag2_num_atoms = 1067  # for pocket atoms without Hs
# frag2_num_atoms = 24000  # for protein atoms
complex_num_atoms = frag1_num_atoms + frag2_num_atoms

split = args.split  # args.split keep the name of splitting method.
if args.clust_file or args.test_core:
  split = None

pdbbind_tasks, pdbbind_datasets, transformers = dc.molnet.load_pdbbind(
    reload=args.reload,
    featurizer="atomic",
    version=args.version,
    frag1_num_atoms=frag1_num_atoms,
    frag2_num_atoms=frag2_num_atoms,
    shard_size=1024,
    split=split,
    split_seed=args.seed,
    clust_file=args.clust_file,
    shuf_labels=args.shuf_labels,
    split_complex=args.split_complex,
    same_protein=args.same_protein,
    same_ligand=args.same_ligand,
    subset=args.subset,
    load_binding_pocket=True,
    data_dir=args.data_dir,
    save_dir=args.save_dir,
    save_timestamp=args.timestamp,
    transform=args.trans,
)

if args.test_core is not None:
  pdbbind_tasks, (test_dataset, _, _), transformers = dc.molnet.load_pdbbind(
      reload=args.reload,
      featurizer="atomic",
      version=args.test_core,
      frag1_num_atoms=frag1_num_atoms,
      frag2_num_atoms=frag2_num_atoms,
      split=None,
      split_seed=args.seed,
      clust_file=args.clust_file,
      split_complex=args.split_complex,
      same_protein=args.same_protein,
      same_ligand=args.same_ligand,
      subset='core',
      load_binding_pocket=True,
      data_dir=args.data_dir,
      save_dir=args.save_dir,
      save_timestamp=args.timestamp,
      transform=args.trans,
  )

if args.feat_only:
  raise SystemExit(0)

if split is not None:
  train_dataset, valid_dataset, test_dataset = pdbbind_datasets
else:
  dataset, _, _ = pdbbind_datasets
  dataset_ids = dataset.ids
  dataset_ids = dataset_ids.tolist()
  if args.test_core is None:
    test_ids = set([])
  else:
    test_ids = set(test_dataset.ids)

  keep_inds = []
  if args.clust_file is None:
    for i, id_ in enumerate(dataset_ids):
      if id_ in test_ids:
        continue
      else:
        keep_inds.append(i)
    random.shuffle(keep_inds)
    # print(keep_inds)
  else:
    with open(args.clust_file) as f:
      clusters = json.load(f)

    clusters_inds = []
    for clust in clusters:
      if all([(id_ not in test_ids) for id_ in clust]):
        clust_inds = []
        for id_ in clust:
          try:
            idx = dataset_ids.index(id_)
            clust_inds.append(idx)
          except ValueError:
            continue
        if clust_inds:
          clusters_inds.append(clust_inds)
    # shuffle clusters.
    # make test set different with different random seeds.
    random.shuffle(clusters_inds)
    keep_inds = sum(clusters_inds, [])
    print(
        f"biggest cluster: {len(max(clusters_inds, key=len))}, total samples: {len(keep_inds)}"
    )

  keep_inds = np.array(keep_inds)
  print(f"keep {len(keep_inds)}/{len(dataset_ids)} in dataset")

  if args.test_core is None:
    N = len(keep_inds)
    N_train = int(0.8 * N)
    N_test = int(0.1 * N)
    train_inds = keep_inds[:N_train]
    valid_inds = keep_inds[N_train:-N_test]
    test_inds = keep_inds[-N_test:]
    # indices will be sorted in dataset.select()
    # so training set will keep the order from PDBbind INDEX.
    # we use INDEX_*_data for samples sorted by pK.
    train_dataset = dataset.select(train_inds)
    valid_dataset = dataset.select(valid_inds)
    test_dataset = dataset.select(test_inds)
  else:
    if args.test_core == '2015':
      if args.version == '2015':
        if args.subset == 'refined': max_N = 2048
        if args.subset == 'general_PL': max_N = 8076
      if args.version == '2018':
        if args.subset == 'refined': max_N = 2648
        if args.subset == 'general_PL': max_N = 11687
    if args.test_core == '2018':
      if args.version == '2015':
        if args.subset == 'refined': max_N = 2130
        if args.subset == 'general_PL': max_N = 8110
      if args.version == '2018':
        if args.subset == 'refined': max_N = 2663
        if args.subset == 'general_PL': max_N = 11561
    print("choice {max_N} samples to be train and valid set.")
    keep_inds = np.random.choice(keep_inds, max_N, replace=False)
    N = len(keep_inds)
    N_train = int(0.9 * N)
    train_inds = keep_inds[:N_train]
    valid_inds = keep_inds[N_train:]
    train_dataset = dataset.select(train_inds)
    valid_dataset = dataset.select(valid_inds)

# transformers = [
#     dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
# ]
# for transformer in transformers:
#   train_dataset = transformer.transform(train_dataset)
#   valid_dataset = transformer.transform(valid_dataset)
#   test_dataset = transformer.transform(test_dataset)

metrics = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score),
    dc.metrics.Metric(dc.metrics.mean_absolute_error)
]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# default
# atom_types=[6, 7, 8, 9, 11, 12, 15, 16, 17, 20, 25, 30, 35, 53, -1]
atom_types = [1, 6, 7, 8, 9, 12, 15, 16, 17, 20, 25, 30, 35, 53, -1]

# [[Rc],[Rs], [Re]], Rc is cutoff, Rs is mean, Re is variance.
default_radial = [[
    1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
    9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
], [0.0, 4.0, 8.0], [0.4]]

min_radial = [[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 2.0, 4.0], [0.4]]
model = dc.models.AtomicConvModel(
    batch_size=args.batch_size,
    atom_types=atom_types,
    max_num_neighbors=4,
    radial=min_radial,
    frag1_num_atoms=frag1_num_atoms,
    frag2_num_atoms=frag2_num_atoms,
    complex_num_atoms=complex_num_atoms,
    component=args.component,
    learning_rate=0.001,
    configproto=config,
)

# Fit trained model
print("Fitting model on train dataset")
patience = 0
best_r2 = 0
best_scores = None
train_evaluator = dc.utils.evaluate.Evaluator(model, train_dataset,
                                              transformers)
valid_evaluator = dc.utils.evaluate.Evaluator(model, valid_dataset,
                                              transformers)
test_evaluator = dc.utils.evaluate.Evaluator(model, test_dataset, transformers)


def copy_checkpoint(source, target='best_checkpoint'):
  import os
  from shutil import copyfile
  dirname = os.path.dirname(source)
  if '/' not in target:
    target = os.path.join(dirname, target)
  for item in os.listdir(dirname):
    item = os.path.join(dirname, item)
    name, ext = os.path.splitext(item)
    if name == source:
      copyfile(source + ext, target + ext)
  return target


if args.shuffle:
  deterministic = False
  shuffle_batches = False
else:
  # samples sorted by pK in PDBbind data set.
  # each batch has samples with similar pK.
  # curriculum learning https://doi.org/10.1145/1553374.1553380
  deterministic = True
  shuffle_batches = True

for i in range(args.max_epoch):
  model.fit(
      train_dataset,
      nb_epoch=1,
      deterministic=deterministic,
      shuffle_batches=shuffle_batches)

  print("Evaluating model at {} epoch".format(i + 1))
  valid_scores = valid_evaluator.compute_model_performance(metrics)
  print("Validation scores")
  print(valid_scores, flush=True)
  if np.isnan(valid_scores['pearson_r2_score']):
    break
  if valid_scores['pearson_r2_score'] < best_r2:
    patience += 1
    if patience > args.patience:
      break
  else:
    last_checkpoint = model.get_checkpoints()[-1]
    best_checkpoint = copy_checkpoint(last_checkpoint)
    patience = 0
    best_r2 = valid_scores['pearson_r2_score']
    print('### Better on valid at epoch {}'.format(i + 1))
    test_scores = test_evaluator.compute_model_performance(metrics)
    print("Testing scores")
    print(test_scores)
    print(flush=True)

model.restore(checkpoint=best_checkpoint)
train_scores = train_evaluator.compute_model_performance(
    metrics, csv_out="train.csv")
valid_scores = valid_evaluator.compute_model_performance(
    metrics, csv_out="valid.csv")
test_scores = test_evaluator.compute_model_performance(
    metrics, csv_out="test.csv")

best_scores = {
    'train': train_scores,
    'valid': valid_scores,
    'test': test_scores
}
print('peformances of model best on validation dataset:')
print(json.dumps(best_scores, indent=2))

with open('best_scores.json', 'w') as f:
  data = vars(args)
  data['best_scores'] = best_scores
  json.dump(data, f, indent=2)

with open('splitted_ids.json', 'w') as f:
  data['splitted_ids'] = {
      'train': list(train_dataset.ids),
      'valid': list(valid_dataset.ids),
      'test': list(test_dataset.ids)
  }
  json.dump(data, f, indent=2)

if split is None:
  shutil.rmtree(train_dataset.data_dir)
  shutil.rmtree(valid_dataset.data_dir)
  if args.test_core is None:
    shutil.rmtree(test_dataset.data_dir)

print(f"Elapsed time {dt.now()- start}")
