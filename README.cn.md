---
typora-copy-images-to: ./README.assets
---

# Prim_diffusion

## 准备工作

1. 克隆仓库

   ```shell
   git clone ...
   cd prim_diffusion
   mkdir data ckpt output logs
   ```

2. 下载数据集：

   ```shell
   ACDC_DOWNLOAD_LINK=https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download
   
   wget $ACDC_DOWNLOAD_LINK -O acdc.zip
   mkdir data
   unzip acdc.zip -d data/ACDC/
   ```

3. 预处理数据 -- （转换为 h5 文件格式，划分训练/验证/测试集）

   ```shell
   cd src/task
   python 01preprocess_dataset.py --data_dir ../../data/ACDC/ --output_dir ../../data/ACDC/
   
   python 04split_csv_by_ID.py ../../data/ACDC/quadra/quadra_per_slice_train.csv
   ```

4. **（可选）** 下载预训练 VAE（SD3.5-large）模型

   ```shell
   cd ../../ # 返回项目根目录
   VAE_DOWNLOAD_LINK=https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true
   
   wget $VAE_DOWNLOAD_LINK -O config/vae/diffusion_pytorch_model.safetensors
   ```

## 训练

### DDPM

```shell
cd src/task/

# 默认配置
CUDA_VISIBLE_DEVICES=0 python 10train_shapedm_guidance.py --config ../../config/training/dm_medium.yaml

# 命令行覆盖参数：
CUDA_VISIBLE_DEVICES=0 python 10train_shapedm_guidance.py --config ../../config/training/dm_medium.yaml max_epoches=100 optimizer.lr=0.001
```

### LDM

```shell
cd src/task/

# 默认配置
CUDA_VISIBLE_DEVICES=0 python 11train_shapeldm_guidance.py --config ../../config/training/ldm_medium.yaml

# 命令行覆盖参数：
CUDA_VISIBLE_DEVICES=0 python 11train_shapeldm_guidance.py --config ../../config/training/ldm_medium.yaml max_epoches=100 optimizer.lr=0.001
```

## 推理

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

------

### 推理样例

**ShapeDM Medium 64**

![exp01](./README.assets/exp01.png)

**Conditional ShapeDM Medium 64**

![exp03](./README.assets/exp03.png)

------
