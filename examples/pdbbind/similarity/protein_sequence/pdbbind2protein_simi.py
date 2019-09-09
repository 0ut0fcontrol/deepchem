"""convert PDBbind to fasta format, and then cluster by uclust in usearch.
"""
import re
import json
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime as dt

from Bio import PDB
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from multiprocessing import Pool

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('index', help='PDBbind data index')
parser.add_argument('output', help='output data in json')
parser.add_argument(
    '-d', '--data_dir', required=True, help="PDBbind data dir, like ./v2015")
parser.add_argument(
    '-e',
    '--exe',
    help='usearch exe, default: ./usearch11.0.667_i86linux32',
    default='./usearch11.0.667_i86linux32')
args = parser.parse_args()
start = dt.now()

DATADIR = Path(args.data_dir)


def SeqFromPDBCode(code):
  protein_pdb = DATADIR / code / (code + '_protein.pdb')
  parser = PDB.PDBParser(QUIET=True)
  try:
    protein = parser.get_structure(code, protein_pdb)
  except:
    print('fail to read {}'.format(code))
    return code, None
  ppb = PDB.PPBuilder()
  seqs = []
  for chain in protein.get_chains():
    seqs.extend([i.get_sequence() for i in ppb.build_peptides(chain)])
    seq_str = ''.join([str(i) for i in seqs])
    a = seqs[0].alphabet
    return code, Seq(seq_str, a)


with open(args.index) as f:
  patt = re.compile(r'^(?P<pdb_id>\w{4}).*', re.MULTILINE)
  codes = np.array(re.findall(patt, f.read()))

succeeded_codes = []
seqs = []
Nones = []
p = Pool()
iter_seqs = p.imap(SeqFromPDBCode, codes)
for code, seq in tqdm(iter_seqs, total=len(codes)):
  if seq is not None:
    succeeded_codes.append(code)
    seqs.append(seq)
p.close()
print(f'succeeded {len(seqs)}/{len(codes)}')

fasta_file = args.index + '.fasta'
with open(fasta_file, 'w') as f:
  for code, seq in zip(succeeded_codes, seqs):
    SeqIO.write(SeqRecord(seq, id=code, description=''), f, 'fasta')
print(f'Sequences save at {fasta_file}')


def uclust2clust(filename):
  cluster_mask = []
  codes = []
  with open(filename) as f:
    for line in f:
      if line[0] == "C":
        continue
      fields = line.split()
      cluster_mask.append(int(fields[1]))
      codes.append(fields[8])
  cluster_num = set(cluster_mask)
  cluster_mask = np.array(cluster_mask)
  codes = np.array(codes)
  clusters = []
  for num in cluster_num:
    clust = list(codes[cluster_mask == num])
    clusters.append(clust)
  return clusters


output_names = []
for simi_cutoff in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
  uclust_file = f'{args.output}.simi{simi_cutoff}.uc'
  subprocess.check_call([
      args.exe, '-cluster_fast', fasta_file, '-id',
      str(simi_cutoff), '-uc', uclust_file
  ])
  clusters = uclust2clust(uclust_file)
  json_name = f'{args.output}.simi{simi_cutoff:.1f}.clust.json'
  output_names.append(json_name)
  with open(json_name, 'w') as f:
    json.dump(clusters, f, indent=4)

print("clusters result in files:")
for i in output_names:
  print(i)
print(f'Elapsed time {dt.now() - start}.')
