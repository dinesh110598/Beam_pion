# %%
import numpy as np, h5py
import os
# %%
h5f = h5py.File('Beam_pion_dataset.h5', 'w')
path = ''

for full_name in os.listdir('./txt'):
    fname, _ = full_name.split('.')
    txt_file = open(f'./txt/{fname}.txt', 'r')
    ecal = []
    hb = []
    ho = []
    
    print(f"Writing {full_name}")

    lines = txt_file.readlines()
    for para in range(len(lines)//15):
        ecal1 = []
        ecal_start, ecal_end = para*15, para*15 + 9
        for i in range(ecal_start, ecal_end):
            row_arr = np.asarray([float(w) for w in lines[i].split()])
            ecal1.append(row_arr)
        ecal.append(np.vstack(ecal1))
        
        hb1 = []
        hb_start, hb_end = para*15 + 9, para*15 + 12
        for i in range(hb_start, hb_end):
            row_arr = np.asarray([float(w) for w in lines[i].split()])
            hb1.append(row_arr)
        hb.append(np.vstack(hb1))
        
        ho1 = []
        ho_start, ho_end = para*15 + 12, para*15 + 15
        for i in range(ho_start, ho_end):
            row_arr = np.asarray([float(w) for w in lines[i].split()])
            ho1.append(row_arr)
        ho.append(np.vstack(ho1))
        
    data_dict = {
        "ECAL": np.stack(ecal, 0),
        "HB": np.stack(hb, 0),
        "HO": np.stack(ho, 0),
    }
    txt_file.close()
    
    # Write to a group in hdf5 file
    group = h5f.create_group(fname)
    for name, data in data_dict.items():
        group.create_dataset(name, data=data, compression="gzip")

h5f.close()
# %%
