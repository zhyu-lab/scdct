import torch



# 10X
RNA_PATH = "/scdct/data/rna.h5ad"
ATAC_PATH = "/scdct/data/atac.h5ad"

TIMESTEPS = 1000
RELEASE_TIME = 0
BATCH_SIZE = 32
DIM = 32
weight_rna_to_atac = 1
weight_atac_to_rna = 1
GENA_LR = 1e-4
GENB_LR = 1e-4
TRANSLATOR_LR = 1e-5
ENC_DEC_LR = 0.001
RNA_LR = 0.0007
ATAC_LR = 0.003
AE_LR = 0.0001
LAMBDA_r = 1
LAMBDA_a = 1
PRETRAINING_EPOCHS=100
EPOCHS = 135
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


