
import pandas as pd
import torch
import pandas as pd
import configs
from train import *
fold_idx = 1
seed = 19193
setup_seed(seed)
models = torch.load(f'./paired_RNA_ADT/models.pth')
RNA_data = preprocess_rna_data(configs.RNA_PATH)
ADT_data = preprocess_adt_data(configs.ADT_PATH)
RNA_input_dim, ADT_input_dim= get_input_dimensions(RNA_data, ADT_data)
rna_train_loader, rna_val_loader, rna_test_loader, adt_train_loader, adt_val_loader, adt_test_loader = create_data_loaders(RNA_data, ADT_data)
rna_encoder, adt_encoder, rna_decoder, adt_decoder = get_encoder_decoder(RNA_input_dim, ADT_input_dim)
rna_encoder.load_state_dict(models[2].state_dict())
adt_encoder.load_state_dict(models[3].state_dict())
rna_decoder.load_state_dict(models[4].state_dict())
adt_decoder.load_state_dict(models[5].state_dict())
rna_encoder = rna_encoder.to(configs.DEVICE)
adt_encoder = adt_encoder.to(configs.DEVICE)
rna_decoder = rna_decoder.to(configs.DEVICE)
adt_decoder = adt_decoder.to(configs.DEVICE)

rna_gen = get_model().to(configs.DEVICE)
adt_gen = get_model().to(configs.DEVICE)
rna_gen.load_state_dict(models[0].state_dict())
adt_gen.load_state_dict(models[1].state_dict())

rna_encoder.eval()
adt_encoder.eval()
rna_decoder.eval()
adt_decoder.eval()
rna_gen.eval()
adt_gen.eval()
all_rna_test_pre = []
all_adt_test_pre = []
with torch.no_grad():
    dataloader1 = DataLoader(rna_test_loader, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader2 = DataLoader(adt_test_loader, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Batch size: {batch_size}")
    for idx, (rna_batch, adt_batch) in enumerate(zip(dataloader1, dataloader2)):
        rna_batch = rna_batch.to(configs.DEVICE)
        adt_batch = adt_batch.to(configs.DEVICE)
        zr_test = rna_encoder(rna_batch)
        za_test = adt_encoder(adt_batch)
        print(f"Predicting rna test set")
        zar_test = translate(za_test, rna_gen,num_samples=10, num_steps=500, eta=0.0) # translating ATAC to RNA
        zar_test = zar_test.view(zar_test.size(0),-1)
        print(f"Predicting adt test set")
        zra_test  = translate(zr_test, adt_gen,num_samples=10, num_steps=500, eta=0.0)   # translating RNA to ATAC
        zra_test = zra_test.view(zra_test.size(0), -1)
        rna_test_pre = rna_decoder(zar_test)
        adt_test_pre= adt_decoder(zra_test)
        all_rna_test_pre.append(rna_test_pre.cpu().numpy())
        all_adt_test_pre.append(adt_test_pre.cpu().numpy())

all_rna_test_pre = np.concatenate(all_rna_test_pre, axis=0)
all_adt_test_pre = np.concatenate(all_adt_test_pre, axis=0)
rna_test_pre_df = pd.DataFrame(all_rna_test_pre)
rna_test_pre_df.to_csv(f'./paired_RNA_ADT/predrna.csv', index=False)
adt_test_pre_df = pd.DataFrame(all_adt_test_pre)
adt_test_pre_df.to_csv(f'./paired_RNA_ADT/predadt.csv', index=False)
