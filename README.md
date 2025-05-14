# scDCT

 A conditional di usion-based deep learning model for high- delity single-cell cross-modality translation

## Requirements

* Python 3.10+.

# Installation

## Clone repository

First, download scDCT from GitHub and activate it:

```bash
git clone https://github.com/zhyu-lab/scdct
cd scdct
conda create --name scdct python=3.10
conda activate scdct
```

## Install requirements

```bash
python -m pip install -r requirements.txt
```

# Usage

The paired_RNA_ATAC folder contains code for processing paired RNA and ATAC data.
The unpaired_RNA_ATAC folder contains code for processing unpaired RNA and ADT data.
The paired_RNA_ADT folder contains code for processing paired RNA and ADT data.

## Train the scDCT model

Both the autoencoder and the diffusion model are trained in the train.py file. All parameter settings are located in the configs.py file.
Our code for training the autoencoder refers to [scButterfly](https://github.com/BioX-NKU/scButterfly).

```bash
python train.py 
```

## Sample and translation

The "sampling.py" is used to generate translated data in the latent space and perform reconstruction.

```bash
python sampling.py
```

# Contact

If you have any questions, please contact 12023132086@stu.nxu.edu.cn.
