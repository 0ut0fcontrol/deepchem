"""computing SMAP pocket similarity for PDBbind.
"""
import re
import json
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime as dt


def parse_index(index_file):
  patt = re.compile(r'(?P<pdb_id>\w{4}).*\((?P<lig_name>.*)\)')
  with open(index_file) as f:
    matchs = re.findall(patt, f.read())
  return matchs


def smap(smap_exe, id_i, lig_i, id_j, lig_j):
  cmd = [smap_exe, id_i, id_j, '/tmp/log']
  job = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  log = job.stdout.decode('utf-8')
  patt = re.compile(
      r'p_value= *([0-9.E+-]+)\t.*query_lig= *(.*)\ttarget_lig= *(.*)')
  print(log)
  matchs = re.findall(patt, log)
  lowest_p = 1
  for p_value, query_lig, target_lig in matchs:
    if lig_i in target_lig and lig_j in query_lig:
      p_value = float(p_value)
      if p_value < lowest_p:
        lowest_p = p_value
  print(id_i, id_j, lowest_p)
  return lowest_p


if __name__ == '__main__':
  start = dt.now()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('index', help='PDBbind data index')
  parser.add_argument('output', help='output data in json')
  parser.add_argument(
      '-s',
      '--smap',
      default='smap_comp.sh',
      help='smap exe, default: smap_comp.sh')
  args = parser.parse_args()

  index = parse_index(args.index)
  p = Pool()
  results = []
  for i, (id_i, lig_i) in enumerate(index):
    for j, (id_j, lig_j) in enumerate(index):
      if j <= i: continue
      result = p.apply_async(smap, args=(args.smap, id_i, lig_i, id_j, lig_j))
      results.append((id_i, id_j, result))
  p.close()
  p.join()

  p_values = []
  for id_i, id_j, result in results:
    p_value = result.get()
    p_values.append((id_i, id_j, p_value))

  with open(args.output, 'w') as f:
    json.dump(p_values, f, indent=4)
  print("Total elapsed time: {}".format(dt.now() - start))