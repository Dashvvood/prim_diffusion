import os
import re
from dataclasses import dataclass, replace
from typing import List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

import h5py
import pandas as pd

class QuadraACDCDataset(Dataset):
    def __init__(self, root_dir, h5data="acdc_quadra.h5", metadata="quadra_per_slice.csv", transform=None):
        self.root_dir = root_dir
        self.data = h5py.File(os.path.join(root_dir, h5data), 'r')
        self.meta = pd.read_csv(os.path.join(root_dir, metadata))
        self.transform = transform if transform is not None else lambda x: x
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        slice_path = row.H5path
        slice = self.data[slice_path][:].astype(np.float32)
        return self.transform(slice), row
    
    def get_slice(self, idx):
        row = self.meta.iloc[idx]
        slice_path = row.H5path
        slice = self.data[slice_path][:]
        
        return slice, row
    
class VanillaDataset(Dataset):
    def __init__(self):
        self.data = np.random.random(20)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

@dataclass
class ACDCDatasetItem:
    patient_dir: str
    patient_info: dict
    ed_img_path: str
    es_img_path: str
    ed_gt_path: str
    es_gt_path: str
    
    ed_img: np.ndarray = None
    ed_gt: np.ndarray = None
    es_img: np.ndarray = None
    es_gt: np.ndarray = None

    def copy(self):
        return replace(
            self,
            patient_info=self.patient_info.copy(),  # 深拷贝字典
            ed_img=np.copy(self.ed_img) if self.ed_img is not None else None,
            ed_gt=np.copy(self.ed_gt) if self.ed_gt is not None else None,
            es_img=np.copy(self.es_img) if self.es_img is not None else None,
            es_gt=np.copy(self.es_gt) if self.es_gt is not None else None
        )


class ACDCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_patient_dirs = self._get_all_patient_dirs()
        self.metadata = self._get_all_patient_metadata()
        
        self.use_ed_img = False
        self.use_es_img = False
        self.use_ed_gt = False
        self.use_es_gt = False
        self.use_4d = False
        self.cache_data = False
        
        self.transform = transform if transform is not None else lambda x: x
        
    def _get_all_patient_dirs(self) -> List:
        dirs = os.listdir(self.root_dir)
        return sorted([dir for dir in dirs if re.match(r'patient\d+', dir)])
    
    def _read_patient_cfg(self, patient_dir):
        patient_info = {}
        with open(os.path.join(patient_dir, 'Info.cfg')) as f:
            for line in f:
                key, value = line.strip().split(':')
                patient_info[key] = value
        return patient_info
    
    def _get_all_patient_metadata(self):
        metadata = {}
        for i, patient_dir in enumerate(self.all_patient_dirs):
            patient_path = os.path.join(self.root_dir, patient_dir)
            patient_info = self._read_patient_cfg(patient_path)
            
            ed_id = int(patient_info['ED'])
            es_id = int(patient_info['ES'])
            
            ed_img_path = os.path.join(patient_path, f"{patient_dir}_frame{ed_id:02d}.nii.gz")
            es_img_path = os.path.join(patient_path, f"{patient_dir}_frame{es_id:02d}.nii.gz")
            
            ed_gt_path = os.path.join(patient_path, f"{patient_dir}_frame{ed_id:02d}_gt.nii.gz")
            es_gt_path = os.path.join(patient_path, f"{patient_dir}_frame{es_id:02d}_gt.nii.gz")

            metadata[i] = ACDCDatasetItem(
                patient_dir, 
                patient_info, 
                ed_img_path, 
                es_img_path, 
                ed_gt_path, 
                es_gt_path, 
                None, None, None, None
            )
        return metadata
    
    def __len__(self):
        return len(self.all_patient_dirs)
    
    def __getitem__(self, idx,):
        # assert idx < len(self)
        
        D = self.metadata[idx] if self.cache_data else self.metadata[idx].copy()
            
        if self.use_ed_img and D.ed_img is None:
            D.ed_img = self.transform(nib.load(D.ed_img_path).get_fdata())
        if self.use_ed_gt and D.ed_gt is None:
            D.ed_gt = self.transform(nib.load(D.ed_gt_path).get_fdata())
        if self.use_es_img and D.es_img is None:
            D.es_img = self.transform(nib.load(D.es_img_path).get_fdata())
        if self.use_es_gt and D.es_gt is None:
            D.es_gt = self.transform(nib.load(D.es_gt_path).get_fdata())

        return D
    
    def configure_usage(self, use_ed_img=True, use_es_img=True, use_ed_gt=True, use_es_gt=True, use_4d=False, cache_data=True):
        self.use_ed_img = use_ed_img
        self.use_es_img = use_es_img
        self.use_ed_gt = use_ed_gt
        self.use_es_gt = use_es_gt
        self.use_4d = use_4d
        self.cache_data = cache_data
        return None

        