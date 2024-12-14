# %%
import numpy as np, matplotlib.pyplot as plt
import h5py, re
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
# %%
def extract_numbers_regex(string):
    return re.findall(r'\d+', string)[0]

def write_to_file():
    with h5py.File('Data/Beam_pion_dataset.h5', 'r') as f:
        eb_list = []
        hb_list = []
        ho_list = []
        energy_list = []

        for g_name in list(f.keys())[:-1]:
            energy = np.log(int(extract_numbers_regex(g_name)))
            eb = np.asarray(f[g_name]['ECAL'])
            hb = np.asarray(f[g_name]['HB'])
            ho = np.asarray(f[g_name]['HO'])

            # Standardizing the values
            eb = np.log(eb + 5.)
            hb = np.log(hb + 10.)
            ho = np.log(ho + 5.)

            eb_list.append(eb)
            hb_list.append(hb)
            ho_list.append(ho)

            energy_list.append(np.full([eb.shape[0]], energy))

        data_dict = {
            "EB": np.concatenate(eb_list, 0),
            "HB": np.concatenate(hb_list, 0),
            "HO": np.concatenate(ho_list, 0),
            "energy": np.concatenate(energy_list, 0),
        }
        shuffle_order = np.random.permutation(data_dict["EB"].shape[0])

        with h5py.File('Data/Normalized_hits.h5', 'w') as g:
            for name, data in data_dict.items():
                g.create_dataset(name, data=data[shuffle_order], compression="gzip")

def get_normalized_hits(min_E=0., max_E=350.):
    with h5py.File('Data/Beam_pion_dataset.h5', 'r') as f:
        eb_list = []
        hb_list = []
        ho_list = []
        energy_list = []

        for g_name in list(f.keys())[:-1]:
            energy = np.log(int(extract_numbers_regex(g_name)))
            # Hits only within the bounds!
            if np.log(min_E + 1e-2) <= energy <= np.log(max_E):
                eb = torch.as_tensor(np.asarray(f[g_name]['ECAL'])).float()
                hb = torch.as_tensor(np.asarray(f[g_name]['HB'])).float()
                ho = torch.as_tensor(np.asarray(f[g_name]['HO'])).float()

                # Standardizing the values
                eb = torch.log(torch.nn.functional.relu(eb) + 1e-3)
                hb = torch.log(torch.nn.functional.relu(hb) + 1e-3)
                ho = torch.log(torch.nn.functional.relu(ho) + 1e-3)
                energy_tensor = torch.full([eb.shape[0]], energy)

                eb_list.append(eb)
                hb_list.append(hb)
                ho_list.append(ho)

                energy_list.append(energy_tensor)

        x_eb = torch.cat(eb_list, 0).float()
        x_hb = torch.cat(hb_list, 0).float()
        x_ho = torch.cat(ho_list, 0).float()
        y = torch.cat(energy_list, 0).float()

    return x_eb, x_hb, x_ho, y

def get_normalized_hits2(min_E=0., max_E=350.):
    with h5py.File('Data/Beam_pion_dataset.h5', 'r') as f:
        dsets = []

        for g_name in list(f.keys())[:-1]:
            energy = np.log(int(extract_numbers_regex(g_name)))
            # Hits only within the bounds!
            if np.log(min_E + 1e-2) <= energy <= np.log(max_E):
                eb = torch.as_tensor(np.asarray(f[g_name]['ECAL'])).float()
                hb = torch.as_tensor(np.asarray(f[g_name]['HB'])).float()
                ho = torch.as_tensor(np.asarray(f[g_name]['HO'])).float()

                # Standardizing the values
                eb = torch.log(torch.nn.functional.relu(eb) + 1e-3)
                hb = torch.log(torch.nn.functional.relu(hb) + 1e-3)
                ho = torch.log(torch.nn.functional.relu(ho) + 1e-3)
                energy_tensor = torch.full([eb.shape[0]], energy)

                dsets.append(TensorDataset(eb, hb, ho, energy_tensor))

    return dsets

def get_normalized_hits_classify(mid_E=15.):
    with h5py.File('Data/Beam_pion_dataset.h5', 'r') as f:
        eb_list = []
        hb_list = []
        ho_list = []
        energy_list = []

        dsets = []
        for g_name in list(f.keys())[:-1]:
            energy = np.log(int(extract_numbers_regex(g_name)))

            eb = torch.as_tensor(np.asarray(f[g_name]['ECAL'])).float()
            hb = torch.as_tensor(np.asarray(f[g_name]['HB'])).float()
            ho = torch.as_tensor(np.asarray(f[g_name]['HO'])).float()

            # Standardizing the values
            eb = torch.log(eb + 5.)
            hb = torch.log(hb + 10.)
            ho = torch.log(ho + 5.)

            eb_list.append(eb)
            hb_list.append(hb)
            ho_list.append(ho)

            if energy <= np.log(mid_E): 
                energy_list.append(torch.full([eb.shape[0]], 0, dtype=torch.int))
            else:
                energy_list.append(torch.full([eb.shape[0]], 1, dtype=torch.int))

        x_eb = torch.cat(eb_list, 0)
        x_hb = torch.cat(hb_list, 0)
        x_ho = torch.cat(ho_list, 0)
        y = torch.cat(energy_list, 0)
        shuffle_order = np.random.permutation(x_eb.shape[0])

    return x_eb[shuffle_order], x_hb[shuffle_order], x_ho[shuffle_order], y[shuffle_order]
    
# %%
class MultiDatasetDataLoader:
    def __init__(self, datasets, batch_size, shuffle=True, **kwargs):
        """
        Custom DataLoader to sample shuffled mini-batches from multiple datasets.

        Args:
            datasets (list of Dataset): List of datasets to load from.
            batch_size (int): Batch size for each dataset.
            shuffle (bool): Whether to shuffle each dataset.
            **kwargs: Additional arguments for individual DataLoaders (e.g., num_workers, drop_last).
        """
        self.loaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
            for dataset in datasets
        ]
        self.iterators = [iter(loader) for loader in self.loaders]

    def __iter__(self):
        """
        Reset iterators for a new epoch and make the loader iterable.
        """
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        """
        Get the next batch from each dataset's DataLoader and combine them.
        """
        combined_batches = []
        for it in self.iterators:
            try:
                batch = next(it)
                combined_batches.append(batch)
            except StopIteration:
                # Handle the case where a dataset runs out of data
                raise StopIteration
        
        # Concatenate the batches along the first dimension
        x_eb, x_hb, x_ho, E = zip(*combined_batches)
        x_eb = torch.cat(x_eb, dim=0)
        x_hb = torch.cat(x_hb, dim=0)
        x_ho = torch.cat(x_ho, dim=0)
        E = torch.cat(E, dim=0)
        
        return x_eb, x_hb, x_ho, E

    def __len__(self):
        """
        Length is defined as the minimum number of batches available from any dataset.
        """
        return min(len(loader) for loader in self.loaders)
# %%
def get_dataloader(batch_size, min_E=0., max_E=350.):    
    dsets = get_normalized_hits2(min_E, max_E)
    return MultiDatasetDataLoader(dsets, batch_size, True)

def get_dataloader_classify(batch_size, min_E=0., max_E=350.):
    x_eb, x_hb, x_ho, y = get_normalized_hits_classify(min_E, max_E)

    dset = TensorDataset(x_eb, x_hb, x_ho, y)
    return DataLoader(dset, batch_size, True)

# %%
def reconstruct(y):
    return np.exp(y)
# # %% Open h5 file
# file = h5py.File('Data/Normalized_hits.h5', 'r')
# # %% Plot each channel in the first observation
# x_ecal = group['ECAL']
# plt.imshow(x_ecal[0])
# # %%
# x_hb = group['HB']
# plt.imshow(x_hb[0])
# # %%
# x_ho = group['HO']
# plt.imshow(x_ho[0])
# # %% Print statistics
# print("ECAL channel")
# print(f"Mean: {np.mean(x_ecal)}, Min: {np.min(group['ECAL'])}, Max: {np.max(group['ECAL'])}")

# print("HB channel")
# print(f"Mean: {np.mean(group['HB'])}, Min: {np.min(group['HB'])}, Max: {np.max(group['HB'])}")

# print("HO channel")
# print(f"Mean: {np.mean(group['HO'])}, Min: {np.min(group['HO'])}, Max: {np.max(group['HO'])}")
# # %% plot histogram
# plt.hist((np.log(np.sum(x_ecal, (1,2)) + 2.0) - 2.5)/5)
# # plt.show()
# # %%
# plt.hist(np.log(np.reshape(np.asarray(group['HB']) - np.min(group['HB']) + 1e-2, (-1,))))
# plt.show()

# plt.hist(np.log(np.reshape(np.asarray(group['HO']) - np.min(group['HO']) + 1e-2, (-1,))))
# plt.show()
# %%
