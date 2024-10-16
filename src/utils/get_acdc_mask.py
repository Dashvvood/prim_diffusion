from dataset import ACDCDataset
import argparse
import imageio
import numpy as np
import os
import motti

def get_acdc_mask(input_folder, output_folder):
    dataset = ACDCDataset(input_folder)
    dataset.configure_usage(
        use_ed_img=False, 
        use_es_img=False, 
        use_ed_gt=True, 
        use_es_gt=True,
        cache_data=False,
    )
    for i in range(len(dataset)):
        
        it = dataset[i]
        patient_dir = it.patient_dir
        
        if it.ed_gt is not None:
            max_idx =  np.argmax(np.sum(it.ed_gt, axis=(0,1)))
            mask_ed = it.ed_gt[..., max_idx]
            mask_ed = motti.normalize_image(mask_ed, target_max=255,)
            output_path = os.path.join(output_folder, f"{patient_dir}_ed_{i:1d}.png")
            imageio.imwrite(output_path, mask_ed.astype(np.uint8))
        
        if it.es_gt is not  None:
            max_idx =  np.argmax(np.sum(it.es_gt, axis=(0,1)))
            mask_es = it.es_gt[..., max_idx]
            mask_es = motti.normalize_image(mask_es, target_max=255)
        
            output_path = os.path.join(output_folder, f"{patient_dir}_es_{i:1d}.png")
            imageio.imwrite(output_path, mask_es.astype(np.uint8))
            
    return dataset
    
def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output folders.")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Path to the input folder')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    
    get_acdc_mask(input_folder, output_folder)
    