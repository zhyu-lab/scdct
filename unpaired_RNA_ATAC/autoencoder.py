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
matplotlib.use('Agg')

def TFIDF(count_mat):
    """
    TF-IDF transformation for matrix.
    """
    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat, axis=0), (count_mat.shape[0], 1))
    nfreqs = count_mat / divide_title
    multiply_title = np.tile(np.log(1 + count_mat.shape[1] / np.sum(count_mat, axis=1)).reshape(-1, 1), (1, count_mat.shape[1]))
    tfidf_mat = sp.csr_matrix(nfreqs.multiply(multiply_title)).T
    return tfidf_mat, divide_title, multiply_title

def get_input_dimensions(RNA_data, ATAC_data):
    RNA_input_dim = len([i for i in RNA_data.var['highly_variable'] if i])
    ATAC_input_dim = ATAC_data.X.shape[1]
    chrom_list = calculate_chrom_list_from_h5ad(ATAC_data)
    return RNA_input_dim, ATAC_input_dim, chrom_list


def RNA_data_preprocessing(RNA_data, normalize_total=True, log1p=True, use_hvg=True, n_top_genes=3000):
    RNA_data_processed = RNA_data.copy()
    RNA_data.var_names_make_unique()
    if normalize_total:
        sc.pp.normalize_total(RNA_data_processed)

    if log1p:
        sc.pp.log1p(RNA_data_processed)

    if use_hvg:
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var['highly_variable']]

    return RNA_data_processed

def ATAC_data_preprocessing(ATAC_data, binary_data=True, filter_features=True, fpeaks=0.005, tfidf=True, normalize=True):
    ATAC_data_processed = ATAC_data.copy()
    if binary_data:
        epi.pp.binarize(ATAC_data_processed)

    if filter_features:
        epi.pp.filter_features(ATAC_data_processed, min_cells=np.ceil(fpeaks * ATAC_data.shape[0]))


    if tfidf:
        count_mat = ATAC_data_processed.X.copy()
        ATAC_data_processed.X, divide_title, multiply_title = TFIDF(count_mat)

    if normalize:
        max_temp = np.max(ATAC_data_processed.X)
        ATAC_data_processed.X = ATAC_data_processed.X / max_temp

    return ATAC_data_processed, divide_title, multiply_title, max_temp


def preprocess_atac_data(file_path):

    ATAC_data = sc.read_h5ad(file_path)
    ATAC_data.obs.index = pd.Series([str(i) for i in range(len(ATAC_data.obs.index))])
    ATAC_data = ATAC_data_preprocessing(ATAC_data)[0]

    return ATAC_data




def calculate_chrom_list_from_h5ad(adata):
    chrom_list = []
    last_one = ''
    for chrom in adata.var['chrom']:
        if chrom.startswith('chr'):
            if chrom != last_one:
                chrom_list.append(1)
                last_one = chrom
            else:
                chrom_list[-1] += 1
        else:
            chrom_list[-1] += 1
    print(f"Chrom list: {chrom_list}")
    return chrom_list



def preprocess_rna_data(file_path):
    # Step 1: Read RNA data
    RNA_data = sc.read_h5ad(file_path)
    RNA_data.obs.index = pd.Series([str(i) for i in range(len(RNA_data.obs.index))])

    # Step 2: Preprocess the data
    RNA_data = RNA_data_preprocessing(RNA_data)

    return RNA_data


def unpaired_split_dataset(
        RNA_data,
        ATAC_data,
        seed=19193
):
    """
    Split datasets into train, validation and test part using different cell types.

    Parameters
    ----------
    RNA_data
        full RNA data for spliting.

    ATAC_data
        full ATAC data for spliting.

    seed
        random seed use to split datasets, if don't give random seed, set it None.

    """

    if not seed is None:
        setup_seed(seed)

    RNA_types = list(RNA_data.obs.cell_type.value_counts().index)
    ATAC_types = list(ATAC_data.obs.cell_type.value_counts().index)
    cell_type_list = list(set(RNA_types) & set(ATAC_types))
    cell_type_r_only = list(set(RNA_types) - set(ATAC_types))
    cell_type_a_only = list(set(ATAC_types) - set(RNA_types))
    celltype_distribution_r = RNA_data.obs.cell_type.value_counts(normalize=True)
    celltype_distribution_a = ATAC_data.obs.cell_type.value_counts(normalize=True)
    celltype_distribution = {}
    for each in cell_type_list:
        celltype_distribution[each] = (celltype_distribution_r[each] + celltype_distribution_a[each]) / 2
    sum_celltype = sum(celltype_distribution.values())
    for each in cell_type_list:
        celltype_distribution[each] /= sum_celltype

    id_list = [[[] for j in range(6)] for i in range(5)]

    RNA_id_list = []
    ATAC_id_list = []
    RNA_id_list_r_only = []
    ATAC_id_list_a_only = []
    for i in range(RNA_data.X.shape[0]):
        if RNA_data.obs.cell_type[i] in cell_type_list:
            RNA_id_list.append(i)
        if RNA_data.obs.cell_type[i] in cell_type_r_only:
            RNA_id_list_r_only.append(i)
    for i in range(ATAC_data.X.shape[0]):
        if ATAC_data.obs.cell_type[i] in cell_type_list:
            ATAC_id_list.append(i)
        if ATAC_data.obs.cell_type[i] in cell_type_a_only:
            ATAC_id_list_a_only.append(i)

    RNA_test_count = int(0.2 * len(RNA_id_list))
    RNA_validation_count = int(0.16 * len(RNA_id_list))
    ATAC_test_count = int(0.2 * len(ATAC_id_list))
    ATAC_validation_count = int(0.16 * len(ATAC_id_list))
    RNA_train_count = int(0.64 * len(ATAC_id_list))
    ATAC_train_count = int(0.64 * len(ATAC_id_list))

    max_train_count = (RNA_train_count + ATAC_train_count) / 2
    max_validation_count = (RNA_validation_count + ATAC_validation_count) / 2

    random.shuffle(RNA_id_list)
    random.shuffle(ATAC_id_list)
    random.shuffle(RNA_id_list_r_only)
    random.shuffle(ATAC_id_list_a_only)

    for i in range(5):
        train_batch_r = RNA_id_list[RNA_test_count + RNA_validation_count:]
        validation_batch_r = RNA_id_list[RNA_test_count:RNA_test_count + RNA_validation_count]
        test_batch_r = RNA_id_list[0:RNA_test_count]
        train_batch_a = ATAC_id_list[ATAC_test_count + ATAC_validation_count:]
        validation_batch_a = ATAC_id_list[ATAC_test_count:ATAC_test_count + ATAC_validation_count]
        test_batch_a = ATAC_id_list[0:ATAC_test_count]
        RNA_id_list.extend(test_batch_r)
        RNA_id_list = RNA_id_list[RNA_test_count:]
        ATAC_id_list.extend(test_batch_a)
        ATAC_id_list = ATAC_id_list[ATAC_test_count:]

        id_list[i][4].extend(test_batch_r)

        id_list[i][5].extend(test_batch_a)

        train_RNA = RNA_data.obs.iloc[train_batch_r, :]
        train_ATAC = ATAC_data.obs.iloc[train_batch_a, :]
        validation_RNA = RNA_data.obs.iloc[validation_batch_r, :]
        validation_ATAC = ATAC_data.obs.iloc[validation_batch_a, :]

        for ctype in cell_type_list:
            train_count = int(max_train_count * celltype_distribution[ctype])
            validation_count = int(max_validation_count * celltype_distribution[ctype])

            id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            id_list[i][0].extend(np.random.choice(id_temp, train_count, replace=True))
            id_temp = list(train_ATAC[train_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][1].extend(np.random.choice(id_temp, train_count, replace=True))
            id_temp = list(validation_RNA[validation_RNA.cell_type == ctype].index.astype(int))
            id_list[i][2].extend(np.random.choice(id_temp, validation_count, replace=True))
            id_temp = list(validation_ATAC[validation_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][3].extend(np.random.choice(id_temp, validation_count, replace=True))

    RNA_test_count = int(0.2 * len(RNA_id_list_r_only))
    ATAC_test_count = int(0.2 * len(ATAC_id_list_a_only))

    for i in range(5):
        test_batch_r = RNA_id_list_r_only[0:RNA_test_count]
        test_batch_a = ATAC_id_list_a_only[0:ATAC_test_count]
        RNA_id_list_r_only.extend(test_batch_r)
        RNA_id_list_r_only = RNA_id_list_r_only[RNA_test_count:]
        ATAC_id_list_a_only.extend(test_batch_a)
        ATAC_id_list_a_only = ATAC_id_list_a_only[ATAC_test_count:]

        id_list[i][4].extend(test_batch_r)

        id_list[i][5].extend(test_batch_a)
    return id_list


def save_dataset_ids(RNA_data, ATAC_data, train_id_r, train_id_a, validation_id_r, validation_id_a, test_id_r, test_id_a, fold_idx):

    pd.DataFrame({
        "train_ids": RNA_data.obs_names[train_id_r]
    }).to_csv(f'./unpaired_RNA_ATAC/train_ids.csv',
              index=False)

    pd.DataFrame({
        "val_ids": RNA_data.obs_names[validation_id_r]
    }).to_csv(f'./unpaired_RNA_ATAC/val_ids.csv',
              index=False)

    pd.DataFrame({
        "test_ids": RNA_data.obs_names[test_id_r]
    }).to_csv(f'./unpaired_RNA_ATAC/test_ids.csv',
              index=False)

    pd.DataFrame({
        "train_ids": ATAC_data.obs_names[train_id_a]
    }).to_csv(f'./unpaired_RNA_ATAC/atac_train_ids.csv',
              index=False)

    pd.DataFrame({
        "val_ids": ATAC_data.obs_names[validation_id_a]
    }).to_csv(f'./unpaired_RNA_ATAC/atac_val_ids.csv',
              index=False)

    pd.DataFrame({
        "test_ids": ATAC_data.obs_names[test_id_a]
    }).to_csv(f'./unpaired_RNA_ATAC/atac_test_ids.csv',
              index=False)


def convert_to_tensors(data_train, data_val, data_test):
    """
    Convert the AnnData object to PyTorch tensors.
    """
    tensor_train = torch.tensor(data_train.X.toarray(), dtype=torch.float32).to(configs.DEVICE)
    tensor_val = torch.tensor(data_val.X.toarray(), dtype=torch.float32).to(configs.DEVICE)
    tensor_test = torch.tensor(data_test.X.toarray(), dtype=torch.float32).to(configs.DEVICE)

    return tensor_train, tensor_val, tensor_test

def convert_to_test_tensors(data_test):
    """
    Convert the AnnData object to PyTorch tensors.
    """

    tensor_test = torch.tensor(data_test.X.toarray(), dtype=torch.float32).to(configs.DEVICE)

    return tensor_test


def get_encoder_decoder(RNA_input_dim, ATAC_input_dim, chrom_list):
    rna_encoder = NetBlock(
        nlayer=2,
        dim_list=[RNA_input_dim, 256, 128],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout_rate=0.1,
        noise_rate=0.5
    )

    atac_encoder = Split_Chrom_Encoder_block(
        nlayer=2,
        dim_list=[ATAC_input_dim, 32 * len(chrom_list), 128],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        chrom_list=chrom_list,
        dropout_rate=0.1,
        noise_rate=0
    )

    rna_decoder = NetBlock(
        nlayer=2,
        dim_list=[128, 256, RNA_input_dim],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        dropout_rate=0.1,
        noise_rate=0
    )

    atac_decoder = Split_Chrom_Decoder_block(
        nlayer=2,
        dim_list=[128, 32 * len(chrom_list), ATAC_input_dim],
        act_list=[nn.LeakyReLU(), nn.Sigmoid()],
        chrom_list=chrom_list,
        dropout_rate=0.1,
        noise_rate=0
    )

    return rna_encoder, atac_encoder, rna_decoder, atac_decoder

def train_ed(rna_encoder, atac_encoder, rna_decoder, atac_decoder,
             rna_tensor_train, atac_tensor_train,
             rna_tensor_val, atac_tensor_val,
             optimizer_rna, optimizer_atac,
             R2R_train_epoch=100, A2A_train_epoch=100,
             batch_size=configs.BATCH_SIZE, r_loss_fn=nn.MSELoss(),
             a_loss_fn=nn.BCELoss(),
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
            torch.save(rna_encoder.state_dict(),
                       f'./unpaired_RNA_ATAC/best_rna_encoder.pth')
            torch.save(rna_decoder.state_dict(),
                       f'./unpaired_RNA_ATAC/best_rna_decoder.pth')
            print("Validation loss decreased, saving model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f'Early stopping triggered.')
                break

    rna_encoder.load_state_dict(torch.load(
        f'./unpaired_RNA_ATAC/best_rna_encoder.pth'))
    rna_decoder.load_state_dict(torch.load(
        f'./unpaired_RNA_ATAC/best_rna_decoder.pth'))

    # ATAC-AE training
    atac_train_losses = []
    atac_val_losses = []
    print(f'Fold {fold_idx} - Starting ATAC-AE training...')
    best_val_loss, patience_counter = float('inf'), 0

    for epoch in range(A2A_train_epoch):
        atac_encoder.train()
        atac_decoder.train()
        train_loss = 0
        dataloader = DataLoader(atac_tensor_train, batch_size=batch_size, shuffle=True, drop_last=False)
        for idx, atac_batch in enumerate(dataloader):
            atac_batch = atac_batch.to(configs.DEVICE)
            optimizer_atac.zero_grad()
            encoded = atac_encoder(atac_batch)
            decoded = atac_decoder(encoded)
            loss = a_loss_fn(decoded, atac_batch)
            loss.backward()
            optimizer_atac.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / np.ceil(atac_tensor_train.size(0) / batch_size)
        atac_train_losses.append(avg_train_loss)
        atac_encoder.eval()
        atac_decoder.eval()
        val_loss = 0
        with torch.no_grad():
            dataloader = DataLoader(atac_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
            for idx, atac_batch in enumerate(dataloader):
                atac_batch = atac_batch.to(configs.DEVICE)
                encoded = atac_encoder(atac_batch)
                decoded = atac_decoder(encoded)
                loss = a_loss_fn(decoded, atac_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / np.ceil(atac_tensor_val.size(0) / batch_size)
        atac_val_losses.append(avg_val_loss)

        print(f'Fold {fold_idx}, Epoch [{epoch + 1}/{A2A_train_epoch}], ATAC Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(atac_encoder.state_dict(),
                       f'./unpaired_RNA_ATAC/best_atac_encoder.pth')
            torch.save(atac_decoder.state_dict(),
                       f'./unpaired_RNA_ATAC/best_atac_decoder.pth')
            print("Validation loss decreased, saving model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve for {patience_counter} epochs.")
            if patience_counter >= patience:
                print(f'Early stopping triggered.')
                break

    atac_encoder.load_state_dict(torch.load(
        f'./unpaired_RNA_ATAC/best_atac_encoder.pth'))
    atac_decoder.load_state_dict(torch.load(
        f'./unpaired_RNA_ATAC/best_atac_decoder.pth'))
    train_loss = np.mean(rna_train_losses) + np.mean(atac_train_losses)
    val_loss = np.mean(rna_val_losses) + np.mean(atac_val_losses)

    print(f'training for fold {fold_idx} complete.')
    return train_loss, val_loss, rna_train_losses, rna_val_losses, atac_train_losses, atac_val_losses


def train_ed_enhanced(ep, rna_gen, atac_gen, rna_encoder, atac_encoder, rna_decoder, atac_decoder,
             rna_tensor_train, atac_tensor_train,
             rna_tensor_val, atac_tensor_val,
             optimizer, batch_size=configs.BATCH_SIZE,
             r_loss_fn=nn.MSELoss(), a_loss_fn=nn.BCELoss()):

    print(f'Epoch {ep} - Starting AE fine-training...')

    num_cells = rna_tensor_train.size(0)
    indices = np.arange(num_cells)
    np.random.shuffle(indices)
    rna_tensor_train = rna_tensor_train[indices]
    atac_tensor_train = atac_tensor_train[indices]

    rna_gen.eval()
    atac_gen.eval()
    rna_encoder.train()
    rna_decoder.train()
    atac_encoder.train()
    atac_decoder.train()
    train_loss = 0
    #batches_sel = 1

    dataloader1 = DataLoader(rna_tensor_train, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader2 = DataLoader(atac_tensor_train, batch_size=batch_size, shuffle=False, drop_last=False)
    for idx, (rna_batch, atac_batch) in enumerate(zip(dataloader1, dataloader2)):
        #if idx > batches_sel-1:
        #    break
        rna_batch = rna_batch.to(configs.DEVICE)
        atac_batch = atac_batch.to(configs.DEVICE)
        optimizer.zero_grad()

        zr_0 = rna_encoder(rna_batch)
        za_0 = atac_encoder(atac_batch)

        with torch.no_grad():
            zra_0 = translate(zr_0, atac_gen).to(configs.DEVICE) # translating RNA to ATAC
            zra_0 = zra_0.view(zra_0.size(0), -1)
            zar_0 = translate(za_0, rna_gen).to(configs.DEVICE) # translating ATAC to RNA
            zar_0 = zar_0.view(zar_0.size(0), -1)

        xr_0 = rna_decoder(zr_0)
        xar_0 = rna_decoder(zar_0)
        xa_0 = atac_decoder(za_0)
        xra_0 = atac_decoder(zra_0)
        loss_rna = r_loss_fn(xar_0, rna_batch)
        print(loss_rna)
        loss_atac =  a_loss_fn(xra_0, atac_batch)
        print(loss_atac)
        loss = loss_rna + loss_atac
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / np.ceil(rna_tensor_train.size(0) / batch_size)

    num_cells = rna_tensor_val.size(0)
    indices = np.arange(num_cells)
    np.random.shuffle(indices)
    rna_tensor_val = rna_tensor_val[indices]
    atac_tensor_val = atac_tensor_val[indices]


    rna_encoder.eval()
    rna_decoder.eval()
    atac_encoder.eval()
    atac_decoder.eval()
    val_loss = 0
    with torch.no_grad():
        dataloader1 = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
        dataloader2 = DataLoader(atac_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
        for idx, (rna_batch, atac_batch) in enumerate(zip(dataloader1, dataloader2)):
            rna_batch = rna_batch.to(configs.DEVICE)
            atac_batch = atac_batch.to(configs.DEVICE)
            zr_0 = rna_encoder(rna_batch)
            za_0 = atac_encoder(atac_batch)
            xr_0 = rna_decoder(zr_0)
            xa_0 = atac_decoder(za_0)

            zra_0 = translate(zr_0, atac_gen).to(configs.DEVICE)
            zra_0 = zra_0.view(zra_0.size(0), -1)
            zar_0 = translate(za_0, rna_gen).to(configs.DEVICE)
            zar_0 = zar_0.view(zar_0.size(0), -1)
            xar_0 = rna_decoder(zar_0)
            xra_0 = atac_decoder(zra_0)
            loss_rna = r_loss_fn(xar_0, rna_batch)
            print(loss_rna)
            loss_atac = a_loss_fn(xra_0, atac_batch)
            print(loss_atac)
            loss = loss_rna + loss_atac
            val_loss += loss.item()

    avg_val_loss = val_loss / np.ceil(rna_tensor_val.size(0) / batch_size)
    return avg_train_loss, avg_val_loss

def evaluate_ae(rna_gen, atac_gen, rna_encoder, atac_encoder, rna_decoder, atac_decoder,
             rna_tensor_val, atac_tensor_val, batch_size=configs.BATCH_SIZE,
             r_loss_fn=nn.MSELoss(), a_loss_fn=nn.BCELoss()):

    rna_gen.eval()
    atac_gen.eval()
    rna_encoder.eval()
    rna_decoder.eval()
    atac_encoder.eval()
    atac_decoder.eval()
    total_loss = 0

    dataloader1 = DataLoader(rna_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader2 = DataLoader(atac_tensor_val, batch_size=batch_size, shuffle=False, drop_last=False)
    for rna_batch, atac_batch in zip(dataloader1, dataloader2):
        rna_batch = rna_batch.to(configs.DEVICE)
        atac_batch = atac_batch.to(configs.DEVICE)

        with torch.no_grad():
            zr_0 = rna_encoder(rna_batch)
            za_0 = atac_encoder(atac_batch)
            xr_0 = rna_decoder(zr_0)
            xa_0 = atac_decoder(za_0)

            zra_0 = translate(zr_0, atac_gen).to(configs.DEVICE)
            zra_0 = zra_0.view(zra_0.size(0), -1)
            zar_0 = translate(za_0, rna_gen).to(configs.DEVICE)
            zar_0 = zar_0.view(zar_0.size(0), -1)
            xar_0 = rna_decoder(zar_0)
            xra_0 = atac_decoder(zra_0)
            loss_rna = r_loss_fn(xar_0, rna_batch)
            loss_atac = a_loss_fn(xra_0, atac_batch)
            loss = loss_rna + loss_atac
            total_loss += loss.item()

    avg_loss = total_loss / np.ceil(rna_tensor_val.size(0) / batch_size)

    return avg_loss


def evaluate_model(rna_encoder, rna_decoder, rna_tensor_test, r_loss_fn, batch_size=configs.BATCH_SIZE):

    rna_encoder.eval()
    rna_decoder.eval()
    test_loss = 0
    with torch.no_grad():
        dataloader1 = DataLoader(rna_tensor_test, batch_size=batch_size, shuffle=False, drop_last=False)
        for rna_batch in dataloader1:
            rna_batch = rna_batch.to(configs.DEVICE)
            encoded = rna_encoder(rna_batch)
            decoded = rna_decoder(encoded)
            loss = r_loss_fn(decoded, rna_batch)
            test_loss += loss.item()
    avg_test_loss = test_loss / np.ceil(rna_tensor_test.size(0) / batch_size)
    print(f'RNA Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss

def evaluate_model_atac(atac_encoder, atac_decoder, atac_tensor_test, a_loss_fn, batch_size=configs.BATCH_SIZE):

    atac_encoder.eval()
    atac_decoder.eval()
    test_loss = 0
    with torch.no_grad():
        dataloader2 = DataLoader(atac_tensor_test, batch_size=batch_size, shuffle=False, drop_last=False)
        for atac_batch in dataloader2:
            atac_batch = atac_batch.to(configs.DEVICE)
            encoded = atac_encoder(atac_batch)
            decoded = atac_decoder(encoded)
            loss = a_loss_fn(decoded, atac_batch)
            test_loss += loss.item()
    avg_test_loss = test_loss / np.ceil(atac_tensor_test.size(0) / batch_size)
    print(f'ATAC Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss



def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


