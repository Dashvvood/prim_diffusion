import os
import re
from dataclasses import dataclass, replace
from typing import List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

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
        
        self.use_ed_img = True
        self.use_es_img = True
        self.use_ed_gt = True
        self.use_es_gt = True
        self.use_4d = False
        self.cache_data = True
        
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
    
    def __getitem__(self, idx):
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
        
        
# class ACDCDatasetMask(ACDCDataset):
#     def __init__(self, root_dir, transform=None):
#         super().__init__(root_dir, transform)
#         self.configure_usage(
#             use_ed_img=False, 
#             use_es_img=False, 
#             use_ed_gt=True, 
#             use_es_gt=True,
#             cache_data=True,
#         )
        
#         self.index2pos = []
        
#         for i in range(len(super())):
#             it = super().__getitem__(i)
#             ed_gt = it.ed_gt
#             es_gt = it.es_gt
            
    
#         self.
        
#     def index2mask(self, idx):
#         pass
        