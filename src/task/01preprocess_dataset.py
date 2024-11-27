"""
Preprocess the ACDC dataset to generate the quadra representation [H, W, C=4] of the masks.
python --data_dir ../../data/ACDC/ --output_dir .
"""

import os
import motti
motti.append_current_dir(os.path.abspath(''))

import h5py
import argparse
import pandas as pd
from tqdm import tqdm
from dataset.ACDC import ACDCDataset
from utils.mask2quadra import mask2quadra


def local_args():
    parser = argparse.ArgumentParser(description='Preprocess ACDC dataset')
    parser.add_argument('--data_dir', type=str, default='data/ACDC', help='Path to the ACDC dataset')
    parser.add_argument('--output_dir', type=str, default='data/ACDC', help='Path to the output directory')
    return parser.parse_args()

headers = ["ID", "Patient", "Phase", "Group", "NbFrame", "FrameIdx", "NbSlice", "SliceIdx", "Height", "Weight", "H5path"]

def data2h5_per_slice(data_dir, output_dir, headers=headers):
    train_set = ACDCDataset(root_dir=os.path.join(data_dir, 'database/training'))
    train_set.configure_usage(
        use_es_gt=True,
        use_ed_gt=True
    )
    
    test_set = ACDCDataset(root_dir=os.path.join(data_dir, 'database/testing'))
    test_set.configure_usage(
        use_es_gt=True,
        use_ed_gt=True
    )
    
    df = pd.DataFrame(columns=headers)
    d = {}
    for i, h in enumerate(headers):
        d[h] =  None
    
    output_dir = os.path.join(output_dir, "quadra")
    os.makedirs(output_dir, exist_ok=True)
    
    n = 0
    with h5py.File(os.path.join(output_dir, "acdc_quadra.h5"), "w") as f:
        grp_name = "quadra_per_slice"
        if grp_name in list(f.keys()):
            grp = f[grp_name]
        else:
            grp = f.create_group(name=grp_name)
        
        for i in tqdm(range(len(train_set)), desc="Train set"):
            patient = train_set[i]
            
            quadra_ed = mask2quadra(patient.ed_gt)
            for j in range(quadra_ed.shape[2]):
                d["ID"] = n
                d["Patient"] = int(patient.patient_dir.strip()[-3:])
                d["Phase"] = "ed"
                d["Group"] = patient.patient_info["Group"].strip()
                d["NbFrame"] = int(patient.patient_info["NbFrame"])
                d["FrameIdx"] = int(patient.patient_info["ED"])
                d["NbSlice"] = quadra_ed.shape[2]
                d["SliceIdx"] = j
                d["Height"] = float(patient.patient_info["Height"])
                d["Weight"] = float(patient.patient_info["Weight"])
                d["H5path"] = f"{grp_name}/{n:04d}"
                df.loc[n] = d
                grp[f"{n:04d}"] = quadra_ed[..., j, :]
                
                n += 1
                
            quadra_es = mask2quadra(patient.es_gt)
            for j in range(quadra_es.shape[2]):
                d["ID"] = n
                d["Patient"] = int(patient.patient_dir.strip()[-3:])
                d["Phase"] = "es"
                d["Group"] = patient.patient_info["Group"].strip()
                d["NbFrame"] = int(patient.patient_info["NbFrame"])
                d["FrameIdx"] = int(patient.patient_info["ES"])
                d["NbSlice"] = quadra_es.shape[2]
                d["SliceIdx"] = j
                d["Height"] = float(patient.patient_info["Height"])
                d["Weight"] = float(patient.patient_info["Weight"])
                d["H5path"] = f"{grp_name}/{n:04d}"
                df.loc[n] = d
                grp[f"{n:04d}"] = quadra_es[..., j, :]
                
                n += 1
        num_train = n
        for i in tqdm(range(len(test_set)), desc="Test set"):
            patient = test_set[i]
            
            quadra_ed = mask2quadra(patient.ed_gt)
            for j in range(quadra_ed.shape[2]):
                d["ID"] = n
                d["Patient"] = int(patient.patient_dir.strip()[-3:])
                d["Phase"] = "ed"
                d["Group"] = patient.patient_info["Group"].strip()
                d["NbFrame"] = int(patient.patient_info["NbFrame"])
                d["FrameIdx"] = int(patient.patient_info["ED"])
                d["NbSlice"] = quadra_ed.shape[2]
                d["SliceIdx"] = j
                d["Height"] = float(patient.patient_info["Height"])
                d["Weight"] = float(patient.patient_info["Weight"])
                d["H5path"] = f"{grp_name}/{n:04d}"
                df.loc[n] = d
                grp[f"{n:04d}"] = quadra_ed[..., j, :]
                
                n += 1
                
            quadra_es = mask2quadra(patient.es_gt)
            for j in range(quadra_es.shape[2]):
                d["ID"] = n
                d["Patient"] = int(patient.patient_dir.strip()[-3:])
                d["Phase"] = "es"
                d["Group"] = patient.patient_info["Group"].strip()
                d["NbFrame"] = int(patient.patient_info["NbFrame"])
                d["FrameIdx"] = int(patient.patient_info["ES"])
                d["NbSlice"] = quadra_es.shape[2]
                d["SliceIdx"] = j
                d["Height"] = float(patient.patient_info["Height"])
                d["Weight"] = float(patient.patient_info["Weight"])
                d["H5path"] = f"{grp_name}/{n:04d}"
                df.loc[n] = d
                grp[f"{n:04d}"] = quadra_es[..., j, :]
                
                n += 1
        num_test = n - num_train
    
    df_train = df.iloc[:num_train].reset_index(drop=True)
    df_test = df.iloc[num_train:].reset_index(drop=True)
    
    df.to_csv(os.path.join(output_dir, "quadra_per_slice_full.csv"), index=False)
    df_train.to_csv(os.path.join(output_dir, "quadra_per_slice_train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "quadra_per_slice_test.csv"), index=False)
    
    return num_train, num_test


if __name__ == '__main__':
    opts = local_args()
    num_train, num_test = data2h5_per_slice(opts.data_dir, opts.output_dir, headers)
    print(f"Number of training samples: {num_train}", f"Number of testing samples: {num_test}")
    
