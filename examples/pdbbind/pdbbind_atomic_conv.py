"""
Script that trains Atomic Conv models on PDBbind dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from deepchem.molnet import load_pdbbind

import argparse

# For stable runs
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=123, help="seed")
parser.add_argument("--subset", required=True)
parser.add_argument("--reload", action='store_true')
parser.add_argument("--trans", action='store_true')
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)

pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind(
    reload=args.reload,
    featurizer="atomic",
    split="random",
    split_seed=seed,
    subset=args.subset,
    load_binding_pocket=True,
    data_dir='/pubhome/jcyang/tmp/deepchem_data_dir',
    save_dir='/tmp',
    save_timestamp=(not args.reload),
    transform=args.trans,
)
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

batch_size = 16
frag1_num_atoms = 70  # for ligand atoms
frag2_num_atoms = 70  # for ligand atoms
# frag2_num_atoms = 24000  # for protein atoms
# frag2_num_atoms = 1000  # for pocket atoms
complex_num_atoms = frag1_num_atoms + frag2_num_atoms
model = dc.models.AtomicConvModel(batch_size=batch_size,
                                  frag1_num_atoms=frag1_num_atoms,
                                  frag2_num_atoms=frag2_num_atoms,
                                  complex_num_atoms=complex_num_atoms)

train_y = train_dataset.y
valid_y = valid_dataset.y
test_y = test_dataset.y

if args.trans:
    for transformer in reversed(transformers):
        if transformer.transform_y:
            train_y = transformer.untransform(train_y)
            valid_y = transformer.untransform(valid_y)
            test_y = transformer.untransform(test_y)


# Fit trained model
print("Fitting model on train dataset")
patience = 0
best = 0
for i in range(100):
  model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)

  print("Evaluating model at {} epoch".format(i + 1))
  train_scores = model.evaluate(train_dataset, [metric], transformers)
  valid_scores = model.evaluate(valid_dataset, [metric], transformers)
  test_scores = model.evaluate(test_dataset, [metric], transformers)

  print("Train scores")
  print(train_scores)

  print("Validation scores")
  print(valid_scores)

  print("Test scores")
  print(test_scores)

  if valid_scores['pearson_r2_score'] < best:
    patience += 1
    if patience > 10:
      break
  else:
    patience = 0
    best = valid_scores['pearson_r2_score']
    train_y_pred = model.predict(train_dataset, transformers)
    valid_y_pred = model.predict(valid_dataset, transformers)
    test_y_pred = model.predict(test_dataset, transformers)

    df = pd.DataFrame({
        '#id': np.ravel(train_dataset.ids),
        'y': np.ravel(train_y),
        'y_pred': np.ravel(train_y_pred)
    })
    df.to_csv(args.subset + '.train.csv', index=False)

    df = pd.DataFrame({
        '#id': np.ravel(valid_dataset.ids),
        'y': np.ravel(valid_y),
        'y_pred': np.ravel(valid_y_pred)
    })
    df.to_csv(args.subset + '.valid.csv', index=False)

    df = pd.DataFrame({
        '#id': np.ravel(test_dataset.ids),
        'y': np.ravel(test_y),
        'y_pred': np.ravel(test_y_pred)
    })
    df.to_csv(args.subset + '.test.csv', index=False)

train_y_pred = model.predict(train_dataset, transformers)
valid_y_pred = model.predict(valid_dataset, transformers)
test_y_pred = model.predict(test_dataset, transformers)

df = pd.DataFrame({
    '#id': np.ravel(train_dataset.ids),
    'y': np.ravel(train_y),
    'y_pred': np.ravel(train_y_pred)
})
df.to_csv(args.subset + '.train.final.csv', index=False)

df = pd.DataFrame({
    '#id': np.ravel(valid_dataset.ids),
    'y': np.ravel(valid_y),
    'y_pred': np.ravel(valid_y_pred)
})
df.to_csv(args.subset + '.valid.final.csv', index=False)

df = pd.DataFrame({
    '#id': np.ravel(test_dataset.ids),
    'y': np.ravel(test_y),
    'y_pred': np.ravel(test_y_pred)
})
df.to_csv(args.subset + '.test.final.csv', index=False)
