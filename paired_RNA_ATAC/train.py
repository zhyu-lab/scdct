import numpy as np
import matplotlib.pyplot as plt
from loaders import *
import torch
from AttnUnet import Unet
from PlainUnet import SimpleUnet_plain
from torch.optim import Adam
import configs
from diffusion import *
import random
import os
from autoencoder import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['TORCH_USE_CUDA_DSA'] = '1'

device = configs.DEVICE
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(plain=False):
    if not plain:
        model = Unet(
            configs.DIM,
            channels=2,
            out_dim=1,
            dim_mults=(1, 2, 4, 8, 8),
        )
    else:
        model = SimpleUnet_plain(
            in_dim=1,
            dim=64,
            out_dim=1,
        )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    return model


def train_with_early_stopping(models, optimizers,
                              data_loaderA_train, data_loaderB_train,
                              data_loaderA_val, data_loaderB_val,
                              data_loaderA_test, data_loaderB_test,
                              num_epochs_ae=100, num_epochs=150, patience=50):
    best_val_loss = float('inf')
    counter = 0

    # # step 1: training AE
    print('training autoencoders......')
    train_ed(models[2], models[3], models[4], models[5],
                 data_loaderA_train, data_loaderB_train,
                 data_loaderA_val, data_loaderB_val,
                 optimizers[2], optimizers[3],
                 R2R_train_epoch=num_epochs_ae, A2A_train_epoch=num_epochs_ae,
                 batch_size=configs.BATCH_SIZE, patience=50)

    # step 2: training translators
    print('training translators started......')
    for i in range(2, 6):
        models[i].eval()
    with torch.no_grad():
        z_r_train = models[2](data_loaderA_train)
        z_r_val = models[2](data_loaderA_val)
        z_r_test = models[2](data_loaderA_test)
        z_a_train = models[3](data_loaderB_train)
        z_a_val = models[3](data_loaderB_val)
        z_a_test = models[3](data_loaderB_test)
    diff_train_losses = []
    diff_val_losses = []
    diff_test_losses = []
    for epoch in range(1, num_epochs + 1):
        avg_train_diff_loss = train_translators(
            epoch, models[0], models[1], optimizers[0], optimizers[1], z_r_train, z_a_train)
        # evaluation of translators
        avg_val_diff_loss = evaluate_translators(models[0], models[1], z_r_val, z_a_val,epoch)
        avg_test_diff_loss = test_translators(models[0], models[1],z_r_test, z_a_test)

        diff_train_losses.append((avg_train_diff_loss))
        diff_val_losses.append(avg_val_diff_loss)
        diff_test_losses.append(avg_test_diff_loss)

        print(f"Epoch {epoch}: Training Diffusion Loss: {avg_train_diff_loss:.4f}")
        print(f"Validation Diffusion Loss: {avg_val_diff_loss:.4f}")
        # Early stopping
        if avg_val_diff_loss < best_val_loss:
            best_val_loss = avg_val_diff_loss
            counter = 0
            fold_idx = 1
            save_dir = f'./paired_RNA_ATAC/'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(models[0], os.path.join(save_dir, 'best_model_genA.pth'))
            torch.save(models[1], os.path.join(save_dir, 'best_model_genB.pth'))
            torch.save(models, os.path.join(save_dir, 'models.pth'))

            print("Validation loss decreased, saving model.")
        else:
            counter += 1
            print(f"Validation loss did not improve for {counter} epochs.")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    models[0] = torch.load(os.path.join(save_dir, 'best_model_genA.pth'))
    models[1] = torch.load(os.path.join(save_dir, 'best_model_genB.pth'))
    return models, diff_train_losses, diff_val_losses


if __name__ == "__main__":
    seed = 19193
    setup_seed(seed)

    fold_idx = 1
    print(f"Training for fold {fold_idx}")
    RNA_data = preprocess_rna_data(configs.RNA_PATH)
    ATAC_data = preprocess_atac_data(configs.ATAC_PATH)

    RNA_input_dim, ATAC_input_dim, chrom_list = get_input_dimensions(RNA_data, ATAC_data)
    rna_train_loader, rna_val_loader, rna_test_loader, atac_train_loader, atac_val_loader, atac_test_loader = create_data_loaders(RNA_data, ATAC_data)
    rna_gen = get_model().to(device)  # noise prediction UNet for RNA
    atac_gen = get_model().to(device)  # noise prediction UNet for ATAC
    rna_encoder, atac_encoder, rna_decoder, atac_decoder = get_encoder_decoder(RNA_input_dim, ATAC_input_dim, chrom_list)
    rna_encoder = rna_encoder.to(device)
    atac_encoder = atac_encoder.to(device)
    rna_decoder = rna_decoder.to(device)
    atac_decoder = atac_decoder.to(device)
    optim_genA = Adam(rna_gen.parameters(), lr=configs.GENA_LR)
    optim_genB = Adam(atac_gen.parameters(), lr=configs.GENB_LR)

    optimizer_rna = Adam(list(rna_encoder.parameters()) + list(rna_decoder.parameters()),
                             lr=configs.RNA_LR)

    optimizer_atac = Adam(list(atac_encoder.parameters()) + list(atac_decoder.parameters()),
                             lr=configs.ATAC_LR)

    optimizer_ae = Adam(list(rna_encoder.parameters()) + list(rna_decoder.parameters()) +
                        list(atac_encoder.parameters()) + list(atac_decoder.parameters()),
                          lr=configs.AE_LR)

    models = [rna_gen, atac_gen, rna_encoder, atac_encoder, rna_decoder, atac_decoder]
    optimizers = [optim_genA, optim_genB, optimizer_rna, optimizer_atac, optimizer_ae]
    best_models,diff_train_losses, diff_val_losses = train_with_early_stopping(
        models, optimizers,
        data_loaderA_train=rna_train_loader,
        data_loaderB_train=atac_train_loader,
        data_loaderA_val=rna_val_loader,
        data_loaderB_val=atac_val_loader,
        data_loaderA_test=rna_test_loader,
        data_loaderB_test=atac_test_loader,
        num_epochs_ae=configs.PRETRAINING_EPOCHS,
        num_epochs=configs.EPOCHS,
        patience=50,
    )




