# %%
import numpy as np, matplotlib.pyplot as plt
import h5py, re
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
# %%
def extract_numbers_regex(string):
    return re.findall(r'\d+', string)[0]

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
                eb = torch.log(F.relu(eb.sum([1,2])) + 1e-3)
                hb = torch.log(F.relu(hb.sum([1,2])) + 1e-3)
                ho = torch.log(F.relu(ho.sum([1,2])) + 1e-3)

                eb_list.append(eb)
                hb_list.append(hb)
                ho_list.append(ho)

                energy_list.append(torch.full([eb.shape[0]], energy))

        x_eb = torch.cat(eb_list, 0).float()
        x_hb = torch.cat(hb_list, 0).float()
        x_ho = torch.cat(ho_list, 0).float()
        y = torch.cat(energy_list, 0).float()
        shuffle_order = np.random.permutation(x_eb.shape[0])

    return x_eb[shuffle_order], x_hb[shuffle_order], x_ho[shuffle_order], y[shuffle_order]