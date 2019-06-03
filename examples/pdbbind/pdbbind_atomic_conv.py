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
import argparse
import numpy as np
import deepchem as dc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--version", default='2015')
parser.add_argument("--subset", default='core')
parser.add_argument("--component", default='binding')
parser.add_argument("--split", default='random')
parser.add_argument("--seed", type=int, default=111)
parser.add_argument("--save_dir", default='/tmp')
parser.add_argument("--data_dir")
parser.add_argument("--reload", action='store_true')
parser.add_argument("--trans", action='store_true')
args = parser.parse_args()

np.random.seed(args.seed)

pdbbind_tasks, pdbbind_datasets, transformers = dc.molnet.load_pdbbind(
    reload=args.reload,
    featurizer="atomic",
    version=args.version,
    split=args.split,
    split_seed=args.seed,
    subset=args.subset,
    load_binding_pocket=True,
    data_dir=args.data_dir,
    save_dir=args.save_dir,
    save_timestamp=(not args.reload),
    transform=args.trans,
)
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

metrics = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score),
    dc.metrics.Metric(dc.metrics.mean_absolute_error)
]

batch_size = 16
frag1_num_atoms = 70  # for ligand atoms
# frag2_num_atoms = 24000  # for protein atoms
frag2_num_atoms = 1000  # for pocket atoms
complex_num_atoms = frag1_num_atoms + frag2_num_atoms
model = dc.models.AtomicConvModel(
    batch_size=batch_size,
    frag1_num_atoms=frag1_num_atoms,
    frag2_num_atoms=frag2_num_atoms,
    complex_num_atoms=complex_num_atoms,
    component=args.component,
)

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
best_r2 = 0
best_scores = None
train_evaluator = dc.utils.evaluate.Evaluator(model, train_dataset,
                                              transformers)
valid_evaluator = dc.utils.evaluate.Evaluator(model, valid_dataset,
                                              transformers)
test_evaluator = dc.utils.evaluate.Evaluator(model, test_dataset, transformers)

for i in range(100):
  model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)

  print("Evaluating model at {} epoch".format(i + 1))
  valid_scores = valid_evaluator.compute_model_performance(metrics)
  print("Validation scores")
  print(valid_scores)

  if valid_scores['pearson_r2_score'] < best_r2:
    patience += 1
    if patience > 10:
      break
  else:
    patience = 0
    best_r2 = valid_scores['pearson_r2_score']
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

with open('best_scores.json', 'w') as f:
  data = vars(args)
  data['best_scores'] = best_scores
  json.dump(data, f, indent=2)
