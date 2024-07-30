# DRR2kVTranslation

This code repository includes a style-transfer module based on a conditional CycleGAN version for generating synthetic kV images for the paper: [Deep-Motion-Net: GNN-based volumetric organ shape reconstruction from single-view 2D projections](https://arxiv.org/abs/2407.06692). The whole code is implemented in [PyTorch](https://pytorch.org/). 

## Getting Started

### Installation Instructions

We recommend that you create a new virtual environment for a clean installation of all relevant dependencies.

```
virtualenv drr2kv
source drr2kv/bin/activate
pip3 install --no-cache-dir torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --no-cache-dir numpy
pip3 install --no-cache-dir Pillow
```

### Dataset Preparation

Unpaired sets of real kilo-voltage (kV) X-ray images (in 8-bit format) and digitally reconstructed radiographs (DRRs) for a given patient are required for model training/validation. The DRRs should be generated with the same field of view (FOV) as the real kV images before start training the model.

### Train

```
python train.py --dataroot *** --model_save_path ***
```

```--dataroot``` => Root folder that exists train/test splits for both datasets (for example, this folder has four sub folders trainDRR/trainkV/testDRR/testkV).

```--model_save_path``` => Folder to store the trained Conditional CycleGAN model instance.

The hyper-parameters can be changed from command.

### Test

To evaluate the model on one example, please use the following command

```
python eval.py --dataroot *** --model_save_path *** --test_results_dir ***
```

```--dataroot``` => Folder that contains a set of DRR images that need to be transferred the style.

```--model_save_path``` => Folder that consists of pretrained Conditional CycleGAN model instance.

```--test_results_dir``` => Folder to store style-transferred images from the model.



