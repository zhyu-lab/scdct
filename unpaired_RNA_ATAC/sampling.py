import anndata
import pandas as pd
import torch
import pandas as pd
import configs
from train import *
from sklearn import metrics
def load_test_ids(fold_idx):
    rna_test_ids = pd.read_csv(f'./unpaired_RNA_ATAC/test_ids.csv')['test_ids'].tolist()
    atac_test_ids = pd.read_csv(f'./unpaired_RNA_ATAC/atac_test_ids.csv')['test_ids'].tolist()
    return rna_test_ids, atac_test_ids

def create_test_data_loaders(RNA_data, ATAC_data, test_id_r, test_id_a):
    RNA_data_test,ATAC_data_test = RNA_data[test_id_r],ATAC_data[test_id_a],
    rna_tensor_test = convert_to_test_tensors(RNA_data_test)
    atac_tensor_test = convert_to_test_tensors(ATAC_data_test)
    return rna_tensor_test, atac_tensor_test


fold_idx = 1
models = torch.load(f'./unpaired_RNA_ATAC/models.pth')
RNA_data = preprocess_rna_data(configs.RNA_PATH)
ATAC_data = preprocess_atac_data(configs.ATAC_PATH)
RNA_input_dim, ATAC_input_dim, chrom_list = get_input_dimensions(RNA_data, ATAC_data)
# original data
test_id_r, test_id_a = load_test_ids(fold_idx)
rna_test_loader, atac_test_loader = create_test_data_loaders(RNA_data, ATAC_data, test_id_r, test_id_a)

rna_encoder, atac_encoder, rna_decoder, atac_decoder = get_encoder_decoder(RNA_input_dim, ATAC_input_dim, chrom_list)
rna_encoder.load_state_dict(models[2].state_dict())
atac_encoder.load_state_dict(models[3].state_dict())
rna_decoder.load_state_dict(models[4].state_dict())
atac_decoder.load_state_dict(models[5].state_dict())
rna_encoder = rna_encoder.to(configs.DEVICE)
atac_encoder = atac_encoder.to(configs.DEVICE)
rna_decoder = rna_decoder.to(configs.DEVICE)
atac_decoder = atac_decoder.to(configs.DEVICE)

rna_gen = get_model().to(configs.DEVICE)
atac_gen = get_model().to(configs.DEVICE)
rna_gen.load_state_dict(models[0].state_dict())
atac_gen.load_state_dict(models[1].state_dict())

rna_encoder.eval()
atac_encoder.eval()
rna_decoder.eval()
atac_decoder.eval()
rna_gen.eval()
atac_gen.eval()
all_rna_test_pre = []
all_atac_test_pre = []
with torch.no_grad():
    dataloader1 = DataLoader(rna_test_loader, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader2 = DataLoader(atac_test_loader, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Batch size: {batch_size}")
    print(('RNA to ATAC predicting...'))
    for idx, rna_batch in enumerate(dataloader1):
        rna_batch = rna_batch.to(configs.DEVICE)
        zr_test = rna_encoder(rna_batch)
        zra_test = translate(zr_test, atac_gen, num_samples=20, num_steps=500, eta=0.0)  # translating RNA to ATAC
        zra_test = zra_test.view(zra_test.size(0), -1)
        atac_test_pre = atac_decoder(zra_test)
        all_atac_test_pre.append(atac_test_pre.cpu().numpy())
    print('ATAC to RNA predicting...')
    for idx, atac_batch in enumerate(dataloader2):
        atac_batch = atac_batch.to(configs.DEVICE)
        za_test = atac_encoder(atac_batch)
        zar_test = translate(za_test, rna_gen, num_samples=20, num_steps=500, eta=0.0)  # translating ATAC to RNA
        zar_test = zar_test.view(zar_test.size(0), -1)
        rna_test_pre = rna_decoder(zar_test)
        all_rna_test_pre.append(rna_test_pre.cpu().numpy())


all_rna_test_pre = np.concatenate(all_rna_test_pre, axis=0)
all_atac_test_pre = np.concatenate(all_atac_test_pre, axis=0)
rna_data_obs = RNA_data.obs
atac_data_obs = ATAC_data.obs

R2A_predict = anndata.AnnData(X=all_atac_test_pre)
R2A_predict.obs = rna_data_obs.iloc[test_id_r, :]
A2R_predict = anndata.AnnData(X=all_rna_test_pre)
A2R_predict.obs = atac_data_obs.iloc[test_id_a, :]
R2A_predict.write_h5ad(f'./unpaired_RNA_ATAC/R2A_predict.h5ad')
A2R_predict.write_h5ad(f'./unpaired_RNA_ATAC/A2R_predict.h5ad')

