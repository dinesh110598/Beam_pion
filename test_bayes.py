# %%
import torch
from torch import nn
import numpy as np, matplotlib.pyplot as plt, h5py
from data_load import get_dataloader, get_normalized_hits
from scipy.stats import gaussian_kde
from concat_net import ConcatNet
# %%
batch_size = 512
width = 96
device = torch.device('cuda:0')
E0, E1 = 1., 15.
# %%
dataloader = get_dataloader(batch_size, min_E=E0, max_E=E1)
net = ConcatNet(width, outputs=2).to(device)
# %%
opt = torch.optim.AdamW(net.parameters())
schedule = torch.optim.lr_scheduler.StepLR(opt, 4)

def NLL_loss(mu, sigma, y):
    return (0.5*torch.log(sigma**2) + (y - mu)**2 / (2*sigma**2)).mean()
# criterion = NLL_loss
criterion = torch.nn.MSELoss()
# %%
net.train()
for ep in range(10):
    for (x_eb, x_hb, x_ho, y_true) in dataloader:
        x_eb, x_hb, x_ho = x_eb.float().to(device), x_hb.float().to(device), x_ho.float().to(device)
        y_true = y_true.float().to(device)

        opt.zero_grad()
        y_pred_theta = net(x_eb, x_hb, x_ho)
        y_pred_mean, y_pred_std = y_pred_theta[:, 0], y_pred_theta[:, 1]
        y_pred = y_pred_mean + torch.randn_like(y_pred_std)*y_pred_std

        loss = criterion(y_pred, y_true)
        loss.backward()
        opt.step()

        print(f"MSE loss: {criterion(y_pred_mean, y_true).item()}")
    schedule.step()
    
net.eval()
# %%
# %% Predict energies of all hits
net2 = net.to(torch.device('cpu'))
x_eb, x_hb, x_ho, y_true = get_normalized_hits(E0, E1)

y_pred = net2(x_eb.float(), x_hb.float(), x_ho.float()).detach()
y_pred = y_pred[:, 0]
# %% Bin all the same energy predictions
y_true2, args = torch.sort(y_true)
y_pred2 = y_pred[args]
lengs = [0]

curr = y_true2[0]
for (i,e) in enumerate(y_true2):
    if np.abs(e - curr) > 0.1:
        curr = e
        lengs.append(i)
lengs.append(len(y_true2))
# %%
y_pred_mean, y_pred_std = y_pred2[:, 0], y_pred2[:, 1]
for (i,j) in zip(lengs[:-1], lengs[1:]):
    E_log = torch.round(torch.mean(y_true[i:j].float())).int()
    E_log_mean, E_log_std = y_pred_mean[i:j].mean(), ((y_pred_std**2).sum()/(j-i+1)**2).sqrt()
# %% Histogram predicted energies for every value of true energy
y_true_exp = y_true2.exp()
y_pred_exp = y_pred2.exp()
for (i,j) in zip(lengs[:-1], lengs[1:]):
    E = torch.round(torch.mean(y_true_exp[i:j].float())).int()
    plt.hist(y_pred_exp[i:j], 100)
    plt.axvline(E, color='black', linestyle='--')
    plt.title(f"E = {E} GeV")
    plt.show()
# %%
