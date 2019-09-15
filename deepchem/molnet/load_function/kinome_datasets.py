"""
PDBBind dataset loader.
"""

from __future__ import division
from __future__ import unicode_literals

import logging
import multiprocessing
import os
import re
import time

import deepchem
import numpy as np
import pandas as pd
import tarfile
from deepchem.feat import rdkit_grid_featurizer as rgf
from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates
from deepchem.feat.atomic_coordinates import SimpleComplexNeighborListFragmentAtomicCoordinates
from deepchem.feat.graph_features import AtomicConvFeaturizer
from deepchem.splits import FingerprintSplitter
from deepchem.splits import Splitter

logger = logging.getLogger(__name__)


def load_kinome(
    reload=True,
    data_dir=None,
    load_binding_pocket=True,
    featurizer="atomic",
    frag1_num_atoms=368,  # from pdbbin
    frag2_num_atoms=1350,
    split=None,
    split_seed=None,
    save_dir=None,
    save_timestamp=False,
):
  # TODO(mapleaf) modify desc
  """Load and featurize raw PDBBind dataset.

    Parameters
    ----------
    ##CHANGE
    reload: Bool, optional
        reload saved featurized and splitted dataset or not
    data_dir: String, optional
        Specifies the data directory to store the original dataset.
    feat: Str
        Either "grid" or "atomic" for grid and atomic featurizations.
    split: Str
        Either "random" or "index"
    split_seed: Int
        Random seed for splitter
    save_dir: String, optional
        Specifies the data directory to store the featurized and splitted dataset.
    """

  #TODO(mapleaf)
  kinome_tasks = ["binding-energy"]
  deepchem_dir = deepchem.utils.get_data_dir()
  if data_dir == None:
    data_dir = deepchem_dir
  data_folder = os.path.join(data_dir, "kinome-wan")

  if save_dir == None:
    save_dir = deepchem_dir
  if load_binding_pocket:
    save_folder = os.path.join(save_dir, "from-kinome-wan",
                               "protein_pocket-%s-%s" % (featurizer, split))
  else:
    save_folder = os.path.join(save_dir, "from-kinome-wan",
                               "full_protein-%s-%s" % (featurizer, split))

  if save_timestamp:
    save_folder = "%s-%s-%s" % (save_folder,
                                time.strftime("%Y%m%d", time.localtime()),
                                re.search(r"\.(.*)", str(time.time())).group(1))
  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      print("Reloaded dataset.")
      return kinome_tasks, all_dataset, transformers
  # TODO(mapleaf): modify
  # dataset_file = os.path.join(data_dir, "pdbbind_v2015.tar.gz")
  # data_folder = os.path.join(data_dir, "v2015")
  # if not os.path.exists(dataset_file):
  #     logger.warning(
  #         "About to download PDBBind full dataset. Large file, 2GB")
  #     deepchem.utils.download_url(
  #         'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
  #         "pdbbind_v2015.tar.gz")
  # if os.path.exists(data_folder):
  #     logger.info("Data directory for %s already exists" % subset)
  # else:
  #     print("Untarring full dataset")
  #     deepchem.utils.untargz_file(dataset_file, dest_dir=data_dir)

  print("\nOriginal dataset:\n%s" % data_folder)
  print("\nFeaturized and splitted dataset:\n%s" % save_folder)

  index_labels_file = os.path.join(data_folder, "kinome-wan.tsv")
  binding_energy = pd.read_table(index_labels_file)

  # Extract labels
  labels = binding_energy.values[:, -17:].flatten()

  # Match complexe names
  ligand_resName_list = [item[-4:-1] for item in binding_energy.columns[-17:]]
  complexe_patterns = [
      # SK090-CLK1-cry-2vag-A-V25-R78
      "%s-.*-.*-.*-%s-[A-Za-z0-9]{3}-%s" % (kinase_sk_no, chain, ligand_resName)
      for kinase_sk_no, chain in binding_energy.values[:, [0, 3]]
      for ligand_resName in ligand_resName_list
  ]
  complexe_names = []
  for i, patt in enumerate(complexe_patterns):
    flag = False
    for complexe_dir in os.listdir(os.path.join(data_folder, patt[-3:])):
      res = re.search(patt, complexe_dir)
      if res != None:
        flag = True
        complexe_names.append(res.group())
        break
    if flag == False:
      complexe_names.append(None)

  # Get indexes missing energy data or pdb files
  miss_energy_data_indexes = [
      i for i, data in enumerate(labels) if np.isnan(data)
  ]
  # solwer than above
  # miss_energy_data_indexes = [
  #     i for i, data_bool in enumerate(binding_energy.isnull().values[:,-17:].flatten())
  #     if data_bool == True
  # ]
  miss_pdb_indexes = [
      i for i, name in enumerate(complexe_names) if name is None
  ]
  # solwer than above
  # miss_pdb_indexes = [
  #     i for i, patt in enumerate(complexe_patterns)
  #     if np.any([
  #         re.search(patt, complexe_dir)
  #         for complexe_dir in os.listdir(os.path.join(data_folder, patt[-3:])
  #             )]) is None
  # ]
  miss_indexes = miss_energy_data_indexes
  [
      miss_indexes.append(i)
      for i in miss_pdb_indexes
      if i not in miss_energy_data_indexes
  ]
  miss_indexes.sort()

  labels = np.delete(labels, miss_indexes)
  complexe_names = np.delete(complexe_names, miss_indexes)

  # Extract locations of data
  complexe_ids = [
      "%s-%s-%s" % (complx[:5], complx[-9], complx[-3:])
      for complx in complexe_names
  ]
  pocket_files = [
      os.path.join(data_folder, complx[-3:], complx, "%s.pocket.pdb" % complx)
      for complx in complexe_names
  ]
  ligand_files = [pocket[:-10] + "ligand.sdf" for pocket in pocket_files]

  max_num_neighbors = 4
  # Cutoff in angstroms
  neighbor_cutoff = 4

  complex_num_atoms = frag1_num_atoms + frag2_num_atoms
  if featurizer == "atomic":
    # Pulled from PDB files. For larger datasets with more PDBs, would use
    # max num atoms instead of exact.
    featurizer = SimpleComplexNeighborListFragmentAtomicCoordinates(
        frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
        neighbor_cutoff)
  else:
    raise ValueError("Featurizer not supported")

  print("\n[%s] Featurizing Complexes for \"%s\" ...\n" % (time.strftime(
      "%Y-%m-%d %H:%M:%S", time.localtime()), data_folder))
  feat_t1 = time.time()
  features, failures = featurizer.featurize_complexes(ligand_files,
                                                      pocket_files)
  feat_t2 = time.time()
  print("\n[%s] Featurization finished, took %0.3f s." % (time.strftime(
      "%Y-%m-%d %H:%M:%S", time.localtime()), feat_t2 - feat_t1))

  # Delete complexe_names, complexe_ids and labels for failing elements
  complexe_names = np.delete(complexe_names, failures)
  complexe_ids = np.delete(complexe_ids, failures)
  labels = np.delete(labels, failures)
  labels = labels.reshape((len(labels), 1))

  print("\n[%s] Construct dataset excluding failing featurization elements..." %
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  dataset = deepchem.data.DiskDataset.from_numpy(
      features, y=labels, ids=complexe_ids, tasks=kinome_tasks)

  # No transformations of data
  transformers = []

  # Split dataset
  print("\n[%s] Split dataset...\n" % time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime()))
  if split == None:
    valid = deepchem.data.DiskDataset.from_numpy([0])
    test = deepchem.data.DiskDataset.from_numpy([0])
    deepchem.utils.save.save_dataset_to_disk(save_folder, dataset, valid, test,
                                             transformers)
    return kinome_tasks, (dataset, None, None), transformers
  elif split == "random":
    splitter = deepchem.splits.RandomSplitter()
  elif split == "ligand":
    ligand_groups = [name[-3:] for name in complexe_ids]
    splitter = deepchem.splits.RandomGroupSplitter(ligand_groups)
  elif split == "kinase":
    kinase_groups = [name[:7] for name in complexe_ids]
    splitter = deepchem.splits.RandomGroupSplitter(kinase_groups)
  else:
    raise ValueError("Splitter for %s not supported" % split)

  # TODO(rbharath): This should be modified to contain a cluster split so
  # structures of the same protein aren't in both train/test
  # splitters = {
  #     'random': deepchem.splits.RandomSplitter(),
  #     'ligand': deepchem.splits.RandomGroupSplitter(ligand_groups),
  #     'kinase': deepchem.splits.RandomGroupSplitter(kinase_groups),
  # }
  # splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset, seed=split_seed)

  all_dataset = (train, valid, test)
  print("\n[%s] Saving dataset to \"%s\" ..." % (time.strftime(
      "%Y-%m-%d %H:%M:%S", time.localtime()), save_folder))
  deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                           transformers)
  return kinome_tasks, all_dataset, transformers