"""
python 04split_csv_by_ID.py ../../data/ACDC/quadra/quadra_per_slice_train.csv 
"""

import argparse
import pandas as pd
import numpy as np
import os

parser = argparse.ArgumentParser(description='Split a CSV file by ID')

parser.add_argument('input', type=str, default="a.csv", help='Input CSV file')
# parser.add_argument('output', type=str, default="output.csv", help='Output CSV file')

opts, _ = parser.parse_known_args()

input_dir = os.path.dirname(opts.input)
input_filename = os.path.splitext(os.path.basename(opts.input))[0]

print(f"Input directory: {input_dir}")
print(f"Input filename without suffix: {input_filename}")

df = pd.read_csv(opts.input)

valsID = []
for group, x in df.groupby("Group"):
    for phase, y in x.groupby("Phase"):
        IDs = np.random.choice(y.ID, 10)
        valsID.extend(IDs)

assert len(valsID) == 100, "Number of validation samples must be 100"

val  = df[df.ID.isin(valsID)]  

train = train = df[~df.ID.isin(valsID)]

train.to_csv(os.path.join(input_dir, input_filename + "_train.csv"), index=False)
val.to_csv(os.path.join(input_dir, input_filename + "_val.csv"), index=False)

print(f"Output files: {input_filename}_train.csv and {input_filename}_val.csv")