import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
from autoencoder import *
import configs

class OriginalDataset(Dataset):
    """
    Dataset class for loading original high-dimensional data, such as RNA and ATAC.
    """
    def __init__(self, data):
        # Assuming the data is already processed and passed as dense numpy array or tensor
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path).values
        # If data is 2D, ensure it has the correct shape
        if self.data.ndim == 2:
            self.data = self.data[:, np.newaxis, :]  # Now data shape is [num_samples, 1, 256]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert to tensor if not already
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample

def create_data_loaders(RNA_data, ADT_data, fold_idx=1):
    # Split dataset into training, validation, and test sets
    id_list = five_fold_split_dataset(RNA_data, ADT_data, seed=19191)
    train_id, validation_id, test_id = id_list[1]
    train_id_r = train_id.copy()
    train_id_a = train_id.copy()
    validation_id_r = validation_id.copy()
    validation_id_a = validation_id.copy()
    test_id_r = test_id.copy()
    test_id_a = test_id.copy()
    save_dataset_ids(RNA_data, ADT_data, train_id_r, train_id_a, validation_id_r, validation_id_a, test_id_r,
                     test_id_a, fold_idx)

    RNA_data_train, RNA_data_val, RNA_data_test = RNA_data[train_id_r], RNA_data[validation_id_r], RNA_data[test_id_r]
    ADT_data_train, ADT_data_val, ADT_data_test = ADT_data[train_id_a], ADT_data[validation_id_a], ADT_data[test_id_a]
    rna_tensor_train, rna_tensor_val, rna_tensor_test = convert_to_tensors(RNA_data_train, RNA_data_val, RNA_data_test)
    adt_tensor_train, adt_tensor_val, adt_tensor_test = convert_to_tensors(ADT_data_train, ADT_data_val,ADT_data_test)

    return rna_tensor_train, rna_tensor_val, rna_tensor_test, adt_tensor_train, adt_tensor_val, adt_tensor_test

