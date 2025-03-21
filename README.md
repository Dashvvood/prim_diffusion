---
typora-copy-images-to: ./README.assets
---

# Prim_diffusion

## Prepare

1. Clone the repository

   ```shell
   git clone ...
   cd prim_diffusion
   mkdir data ckpt output logs
   ```

2. Download Dataset:

   ```shell
   ACDC_DOWNLOAD_LINK=https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download
   
   wget $ACDC_DOWNLOAD_LINK -O acdc.zip
   mkdir data
   unzip acdc.zip -d data/ACDC/
   ```

3. Pre-process -- (convert to h5file format, split train/val/test)

   ```shell
   cd src/task
   python 01preprocess_dataset.py --data_dir ../../data/ACDC/ --output_dir ../../data/ACDC/
   
   python 04split_csv_by_ID.py ../../data/ACDC/quadra/quadra_per_slice_train.csv 
   ```

4. **(Optional)** Download pretrained vae (SD3.5-large) model

   ```shell
   cd ../../ # go back to project root directory
   VAE_DOWNLOAD_LINK=https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true
   
   wget $VAE_DOWNLOAD_LINK -O config/vae/diffusion_pytorch_model.safetensors
   ```

## Training

### DDPM

```shell
cd src/task/

# default setting
CUDA_VISIBLE_DEVICES=0 python 10train_shapedm_guidance.py --config ../../config/training/dm_medium.yaml 

# override by command line:
CUDA_VISIBLE_DEVICES=0 python 10train_shapedm_guidance.py --config ../../config/training/dm_medium.yaml max_epoches=100 optimizer.lr=0.001
```



### LDM

```shell
cd src/task/

# default setting
CUDA_VISIBLE_DEVICES=0 python 11train_shapeldm_guidance.py --config ../../config/training/ldm_medium.yaml 

# override by command line:
CUDA_VISIBLE_DEVICES=0 python 11train_shapeldm_guidance.py --config ../../config/training/ldm_medium.yaml max_epoches=100 optimizer.lr=0.001
```



## Inference

### DDPM

```shell
CUDA_VISIBLE_DEVICES=0 python 07inference_guide.py \
   --ckpt_path XXXX \
   --config_dir ../../config/model/dm_medium/ \
   --batch_size 64 \
   --total_num 1024 \
   --output_type "numpy" \
   --guidance_scale 7.5 \
   --num_inference_steps 1000 \
   --output_dir ../../output/ \
```

### Latent DM
```shell
CUDA_VISIBLE_DEVICES=0 python 07inference_guide.py \
   --ckpt_path XXXX \
   --config_dir ../../config/model/ldm_large/ \
   --batch_size 64 \
   --total_num 1024 \
   --output_type "numpy" \
   --guidance_scale 7.5 \
   --num_inference_steps 1000 \
   --output_dir ../../output/ \
   --latent
```

---

### Inference samples

**ShapeDM Medium 64**

![exp01](./README.assets/exp01.png)

**Conditional ShapeDM Medium 64**

![exp03](./README.assets/exp03.png)

---

---

