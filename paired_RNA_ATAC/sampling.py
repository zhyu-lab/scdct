import pandas as pd
import torch
import pandas as pd
import configs
from train import *
import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch
import configs
from configs import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

device = configs.DEVICE
fold_idx = 1
models = torch.load(f'./paired_RNA_ATAC/models.pth')
RNA_data = preprocess_rna_data(configs.RNA_PATH)
ATAC_data = preprocess_atac_data(configs.ATAC_PATH)
RNA_input_dim, ATAC_input_dim, chrom_list = get_input_dimensions(RNA_data, ATAC_data)
# original data
rna_train_loader, rna_val_loader, rna_test_loader, atac_train_loader, atac_val_loader, atac_test_loader = create_data_loaders(RNA_data, ATAC_data)
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
        for idx, (rna_batch, atac_batch) in enumerate(zip(dataloader1, dataloader2)):
            rna_batch = rna_batch.to(configs.DEVICE)
            atac_batch = atac_batch.to(configs.DEVICE)
            zr_test = rna_encoder(rna_batch)
            za_test = atac_encoder(atac_batch)
            print(f"Predicting rna test set")
            zar_test = translate(za_test, rna_gen,num_samples=10, num_steps=500, eta=0.0) # translating ATAC to RNA
            zar_test = zar_test.view(zar_test.size(0),-1)
            print(f"Predicting atac test set")
            zra_test  = translate(zr_test, atac_gen,num_samples=10, num_steps=500, eta=0.0)   # translating RNA to ATAC
            zra_test = zra_test.view(zra_test.size(0), -1)
            rna_test_pre = rna_decoder(zar_test)
            atac_test_pre= atac_decoder(zra_test)
            all_rna_test_pre.append(rna_test_pre.cpu().numpy())
            all_atac_test_pre.append(atac_test_pre.cpu().numpy())


all_rna_test_pre = np.concatenate(all_rna_test_pre, axis=0)
all_atac_test_pre = np.concatenate(all_atac_test_pre, axis=0)

rna_test_pre_df = pd.DataFrame(all_rna_test_pre)
rna_test_pre_df.to_csv(f'./paired_RNA_ATAC/predrna.csv',index=False)

atac_test_pre_df = pd.DataFrame(all_atac_test_pre)
atac_test_pre_df.to_csv(f'./paired_RNA_ATAC/predatac.csv',index=False)

