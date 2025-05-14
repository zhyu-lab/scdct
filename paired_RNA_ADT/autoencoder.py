from torch.utils.data import DataLoader

from ed import *
import pandas as pd
import torch
import numpy as np
import random
import os
import configs
import scanpy as sc
import episcanpy.api as epi
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
from diffusion import translate
from scipy.sparse import csr_matrix
from scipy.stats.mstats import gmean
import anndata as ad
matplotlib.use('Agg')

def get_input_dimensions(RNA_data, ADT_data):
    RNA_input_dim = len([i for i in RNA_data.var['highly_variable'] if i])
    ADT_input_dim = ADT_data.X.shape[1]

    return RNA_input_dim, ADT_input_dim


def RNA_data_preprocessing(RNA_data, normalize_total=True, log1p=True, use_hvg=True, n_top_genes=3000):
    """
    Preprocessing for RNA data, using scanpy.
    """

    RNA_data.var_names_make_unique()

    if normalize_total:
        sc.pp.normalize_total(RNA_data)

    if log1p:
        sc.pp.log1p(RNA_data)

    if use_hvg:
        sc.pp.highly_variable_genes(RNA_data, n_top_genes=n_top_genes)
        RNA_data = RNA_data[:, RNA_data.var['highly_variable']]

    return RNA_data

def CLR_transform(ADT_data):
    """
    Centered log-ratio transformation for ADT data.

    Parameters
    ----------
    ADT_data: Anndata
        ADT anndata for processing.

    Returns
    ----------
    ADT_data_processed: Anndata
        ADT data with CLR transformation preprocessed.

    gmean_list
        vector of geometric mean for ADT expression of each cell.
    """
    ADT_matrix = ADT_data.X.todense()
    gmean_list = []
    for i in range(ADT_matrix.shape[0]):
        temp = []
        for j in range(ADT_matrix.shape[1]):
            if not ADT_matrix[i, j] == 0:
                temp.append(ADT_matrix[i, j])
        gmean_temp = gmean(temp)
        gmean_list.append(gmean_temp)
        for j in range(ADT_matrix.shape[1]):
            if not ADT_matrix[i, j] == 0:
                ADT_matrix[i, j] = np.log(ADT_matrix[i, j] / gmean_temp)
    ADT_data_processed = ad.AnnData(csr_matrix(ADT_matrix), obs=ADT_data.obs, var=ADT_data.var)
    return ADT_data_processed, gmean_list


def preprocess_adt_data(file_path):

    ADT_data = sc.read_h5ad(file_path)
    ADT_data.X = csr_matrix(ADT_data.X)
    ADT_data  = CLR_transform(ADT_data)[0]
    return ADT_data




def preprocess_rna_data(file_path):
    """
    Preprocessing RNA data and split into train, val, and test sets.
    """
    # Step 1: Read RNA data
    RNA_data = sc.read_h5ad(file_path)
    RNA_data.X = csr_matrix(RNA_data.X)
    # Step 2: Preprocess the data
    RNA_data = RNA_data_preprocessing(RNA_data)

    return RNA_data


def five_fold_split_dataset(
        RNA_data,
        ADT_data,
        seed=19193
):
    if not seed is None:
        setup_seed(seed)

    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)

    id_list = []
    test_count = int(0.2 * len(temp))
    validation_count = int(0.16 * len(temp))

    for i in range(5):
        test_id = temp[: test_count]
        validation_id = temp[test_count: test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        temp.extend(test_id)
        temp = temp[test_count:]

        id_list.append([train_id, validation_id, test_id])

    return id_list

def save_dataset_ids(RNA_data, ADT_data, train_id_r, train_id_a, validation_id_r, validation_id_a, test_id_r,
                     test_id_a, fold_idx):
    pd.DataFrame({
        "train_ids": RNA_data.obs_names[train_id_r]
    }).to_csv(f'./paired_RNA_ADT/train_ids.csv',
              index=False)

    pd.DataFrame({
        "val_ids": RNA_data.obs_names[validation_id_r]
    }).to_csv(f'./paired_RNA_ADT/val_ids.csv',
              index=False)

    pd.DataFrame({
        "test_ids": RNA_data.obs_names[test_id_r]
    }).to_csv(f'./paired_RNA_ADT/test_ids.csv',
              index=False)

    pd.DataFrame({
        "train_ids": ADT_data.obs_names[train_id_a]
    }).to_csv(f'./paired_RNA_ADT/adt_train_ids.csv',
              index=False)

    pd.DataFrame({
        "val_ids": ADT_data.obs_names[validation_id_a]
    }).to_csv(f'./paired_RNA_ADT/adt_val_ids.csv',
              index=False)

    pd.DataFrame({
        "test_ids": ADT_data.obs_names[test_id_a]
    }).to_csv(f'./paired_RNA_ADT/adt_test_ids.csv',
              index=False)

def convert_to_tensors(data_train, data_val, data_test):
    """
    Convert the AnnData object to PyTorch tensors.
    """
    tensor_train = torch.tensor(data_train.X.toarray(), dtype=torch.float32).to(configs.DEVICE)
    tensor_val = torch.tensor(data_val.X.toarray(), dtype=torch.float32).to(configs.DEVICE)
    tensor_test = torch.tensor(data_test.X.toarray(), dtype=torch.float32).to(configs.DEVICE)

    return tensor_train, tensor_val, tensor_test


def save_model(rna_encoder, adt_encoder, rna_decoder, adt_decoder, RNA_input_dim, ADT_input_dim):
    torch.save({
        'rna_encoder_state_dict': rna_encoder.state_dict(),
        'rna_decoder_state_dict': rna_decoder.state_dict(),
        'adt_encoder_state_dict': adt_encoder.state_dict(),
        'adt_decoder_state_dict': adt_decoder.state_dict(),
        'RNA_input_dim': RNA_input_dim,
        'ADT_input_dim': ADT_input_dim
    }, f"./paired_RNA_ADT/model_info.pth")

    print("Model and dimensions saved.")


def get_encoder_decoder(RNA_input_dim, ADT_input_dim):
    rna_encoder = NetBlock(
        nlayer=2,
        dim_list=[RNA_input_dim, 256, 128],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout_rate=0.1,
        noise_rate=0.5
    )

    adt_encoder = NetBlock(
        nlayer=2,
        dim_list=[ADT_input_dim, 128, 128],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout_rate=0.1,
        #noise_rate=0.2
        noise_rate=0
    )

    rna_decoder = NetBlock(
        nlayer=2,
        dim_list=[128, 256, RNA_input_dim],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout_rate=0.1,
        noise_rate=0
    )

    adt_decoder = NetBlock(
        nlayer=2,
        dim_list=[128, 128, ADT_input_dim],
        act_list=[nn.LeakyReLU(), nn.Identity()],
        dropout_rate=0.1,
        noise_rate=0
    )

    return rna_encoder, adt_encoder, rna_decoder, adt_decoder

def train_ed(rna_encoder, adt_encoder, rna_decoder, adt_decoder,
             rna_tensor_train, adt_tensor_train,
             rna_tensor_val, adt_tensor_val,
             optimizer_rna, optimizer_adt,
             R2R_train_epoch=100, A2A_train_epoch=100,
             batch_size=configs.BATCH_SIZE, r_loss_fn=nn.MSELoss(),
             a_loss_fn=nn.MSELoss(),
             patience=20, fold_idx=1):

    # RNA-AE training
    rna_train_losses = []
    rna_val_losses = []
    print(f'Fold {fold_idx} - Starting RNA-AE training...')
    best_val_loss, patience_counter = float('inf'), 0

    for epoch in range(R2R_train_epoch):
        rna_encoder.train()
        rna_decoder.train()
        train_loss = 0
        dataloader = DataLoader(rna_tensor_train, batch_size=batch_size, shuffle=True, drop_last=False)
        for idx, rna_batch in enumerate(dataloader):
            rna_batch = rna_batch.to(configs.DEVICE)

            optimizer_rna.zero_grad()
            encoded = rna_encoder(rna_batch)
            decoded = rna_decoder(encoded)
            loss = r_loss_fn(decoded, rna_batch)
            loss.backward()
            optimizer_rna.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / np.ceil(rna_tensor_train.size(0) / batch_size)
        rna_train_losses.append(avg_train_loss)

        rna_encoder.eval()
        rna_decoder.eval()
        val_loss = 0
        with torch.no_grad():
            dataloader = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
            for idx, rna_batch in enumerate(dataloader):
                rna_batch = rna_batch.to(configs.DEVICE)
                encoded = rna_encoder(rna_batch)
                decoded = rna_decoder(encoded)
                loss = r_loss_fn(decoded, rna_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / np.ceil(rna_tensor_val.size(0) / batch_size)
        rna_val_losses.append(avg_val_loss)

        print(f'Fold {fold_idx}, Epoch [{epoch + 1}/{R2R_train_epoch}], RNA Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(rna_encoder.state_dict(), f'./paired_RNA_ADT/best_rna_encoder.pth')
            torch.save(rna_decoder.state_dict(), f'./paired_RNA_ADT/best_rna_decoder.pth')
            print("Validation loss decreased, saving model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f'Early stopping triggered.')
                break

    rna_encoder.load_state_dict(torch.load(f'./paired_RNA_ADT/best_rna_encoder.pth'))
    rna_decoder.load_state_dict(torch.load(f'./paired_RNA_ADT/best_rna_decoder.pth'))

    # ADT-AE pretraining
    adt_train_losses = []
    adt_val_losses = []
    print(f'Fold {fold_idx} - Starting ADT-AE training...')
    best_val_loss, patience_counter = float('inf'), 0

    for epoch in range(A2A_train_epoch):
        adt_encoder.train()
        adt_decoder.train()
        train_loss = 0
        dataloader = DataLoader(adt_tensor_train, batch_size=batch_size, shuffle=True, drop_last=False)
        for idx, adt_batch in enumerate(dataloader):
            adt_batch = adt_batch.to(configs.DEVICE)
            optimizer_adt.zero_grad()
            encoded = adt_encoder(adt_batch)
            decoded = adt_decoder(encoded)
            loss = a_loss_fn(decoded, adt_batch)
            loss.backward()
            optimizer_adt.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / np.ceil(adt_tensor_train.size(0) / batch_size)
        adt_train_losses.append(avg_train_loss)
        adt_encoder.eval()
        adt_decoder.eval()
        val_loss = 0
        with torch.no_grad():
            dataloader = DataLoader(adt_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
            for idx, adt_batch in enumerate(dataloader):
                adt_batch = adt_batch.to(configs.DEVICE)
                encoded = adt_encoder(adt_batch)
                decoded = adt_decoder(encoded)
                loss = a_loss_fn(decoded, adt_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / np.ceil(adt_tensor_val.size(0) / batch_size)
        adt_val_losses.append(avg_val_loss)

        print(f'Fold {fold_idx}, Epoch [{epoch + 1}/{A2A_train_epoch}], ADT Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(adt_encoder.state_dict(), f'./paired_RNA_ADT/best_adt_encoder.pth')
            torch.save(adt_decoder.state_dict(), f'./paired_RNA_ADT/best_adt_decoder.pth')
            print("Validation loss decreased, saving model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f'Early stopping triggered.')
                break


    adt_encoder.load_state_dict(torch.load(f'./paired_RNA_ADT/best_adt_encoder.pth'))
    adt_decoder.load_state_dict(torch.load(f'./paired_RNA_ADT/best_adt_decoder.pth'))

    train_loss = np.mean(rna_train_losses) + np.mean(adt_train_losses)
    val_loss = np.mean(rna_val_losses) + np.mean(adt_val_losses)

    print(f'Pre-training for fold {fold_idx} complete.')

    return train_loss, val_loss, rna_train_losses, rna_val_losses, adt_train_losses, adt_val_losses


def train_ed_enhanced(ep, rna_gen, adt_gen, rna_encoder, adt_encoder, rna_decoder, adt_decoder,
             rna_tensor_train, adt_tensor_train,
             rna_tensor_val, adt_tensor_val,
             optimizer, batch_size=configs.BATCH_SIZE,
             r_loss_fn=nn.MSELoss(), a_loss_fn=nn.MSELoss()):

    print(f'Epoch {ep} - Starting AE fine-training...')

    num_cells = rna_tensor_train.size(0)
    indices = np.arange(num_cells)
    np.random.shuffle(indices)
    rna_tensor_train = rna_tensor_train[indices]
    adt_tensor_train = adt_tensor_train[indices]

    rna_gen.eval()
    adt_gen.eval()
    rna_encoder.train()
    rna_decoder.train()
    adt_encoder.train()
    adt_decoder.train()
    train_loss = 0
    batches_sel = 1

    dataloader1 = DataLoader(rna_tensor_train, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader2 = DataLoader(adt_tensor_train, batch_size=batch_size, shuffle=False, drop_last=False)
    for idx, (rna_batch, adt_batch) in enumerate(zip(dataloader1, dataloader2)):
        if idx > batches_sel-1:
            break
        rna_batch = rna_batch.to(configs.DEVICE)
        adt_batch = adt_batch.to(configs.DEVICE)
        optimizer.zero_grad()

        zr_0 = rna_encoder(rna_batch)
        za_0 = adt_encoder(adt_batch)

        with torch.no_grad():
            zra_0 = translate(zr_0, adt_gen).to(configs.DEVICE) # translating RNA to ATAC
            zra_0 = zra_0.view(zra_0.size(0), -1)
            zar_0 = translate(za_0, rna_gen).to(configs.DEVICE) # translating ATAC to RNA
            zar_0 = zar_0.view(zar_0.size(0), -1)

        xr_0 = rna_decoder(zr_0)
        xar_0 = rna_decoder(zar_0)
        xa_0 = adt_decoder(za_0)
        xra_0 = adt_decoder(zra_0)

        loss_rna = r_loss_fn(xar_0, rna_batch)
        print(loss_rna)
        loss_adt =  a_loss_fn(xra_0, adt_batch)
        print(loss_adt)
        loss = loss_rna + loss_adt
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / batches_sel

    num_cells = rna_tensor_val.size(0)
    indices = np.arange(num_cells)
    np.random.shuffle(indices)
    rna_tensor_val = rna_tensor_val[indices]
    adt_tensor_val = adt_tensor_val[indices]


    rna_encoder.eval()
    rna_decoder.eval()
    adt_encoder.eval()
    adt_decoder.eval()
    val_loss = 0
    with torch.no_grad():
        dataloader1 = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
        dataloader2 = DataLoader(adt_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
        for idx, (rna_batch, adt_batch) in enumerate(zip(dataloader1, dataloader2)):
            if idx > batches_sel - 1:
                break
            rna_batch = rna_batch.to(configs.DEVICE)
            adt_batch = adt_batch.to(configs.DEVICE)
            zr_0 = rna_encoder(rna_batch)
            za_0 = adt_encoder(adt_batch)
            xr_0 = rna_decoder(zr_0)
            xa_0 = adt_decoder(za_0)

            zra_0 = translate(zr_0, adt_gen).to(configs.DEVICE)
            zra_0 = zra_0.view(zra_0.size(0), -1)
            zar_0 = translate(za_0, rna_gen).to(configs.DEVICE)
            zar_0 = zar_0.view(zar_0.size(0), -1)
            xar_0 = rna_decoder(zar_0)
            xra_0 = adt_decoder(zra_0)
            loss_rna = r_loss_fn(xar_0, rna_batch)
            print(loss_rna)
            loss_adt = a_loss_fn(xra_0, adt_batch)
            print(loss_adt)
            loss = loss_rna + loss_adt
            val_loss += loss.item()

    avg_val_loss = val_loss / batches_sel
    return avg_train_loss, avg_val_loss

def evaluate_ae(rna_gen, adt_gen, rna_encoder, adt_encoder, rna_decoder, adt_decoder,
             rna_tensor_val, adt_tensor_val, batch_size=configs.BATCH_SIZE,
             r_loss_fn=nn.MSELoss(), a_loss_fn=nn.MSELoss()):

    rna_gen.eval()
    adt_gen.eval()
    rna_encoder.eval()
    rna_decoder.eval()
    adt_encoder.eval()
    adt_decoder.eval()
    total_loss = 0

    dataloader1 = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader2 = DataLoader(adt_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
    for rna_batch, adt_batch in zip(dataloader1, dataloader2):
        rna_batch = rna_batch.to(configs.DEVICE)
        adt_batch = adt_batch.to(configs.DEVICE)

        with torch.no_grad():
            zr_0 = rna_encoder(rna_batch)
            za_0 = adt_encoder(adt_batch)
            xr_0 = rna_decoder(zr_0)
            xa_0 = adt_decoder(za_0)

            zra_0 = translate(zr_0, adt_gen).to(configs.DEVICE)
            zra_0 = zra_0.view(zra_0.size(0), -1)
            zar_0 = translate(za_0, rna_gen).to(configs.DEVICE)
            zar_0 = zar_0.view(zar_0.size(0), -1)
            xar_0 = rna_decoder(zar_0)
            xra_0 = adt_decoder(zra_0)
            loss_rna = r_loss_fn(xar_0, rna_batch)
            loss_adt = a_loss_fn(xra_0, adt_batch)
            loss = loss_rna + loss_adt
            total_loss += loss.item()

    avg_loss = total_loss / np.ceil(rna_tensor_val.size(0) / batch_size)

    return avg_loss



def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

