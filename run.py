# %%
import torch
from torch import nn
import numpy as np, matplotlib.pyplot as plt, itertools
from data_load import get_dataloader, get_normalized_hits
from concat_net import ConcatNet, MeanRegularizedLoss
# %% Hyperparameters
batch_size = 128
width = 64
device = torch.device('cuda:0')
E0, E1 = 0., 15.
lambdas = [0., 0.25, 0.5, 0.75, 1.0]
# %%
dataloader = get_dataloader(batch_size, min_E=E0, max_E=E1)
nets = [ConcatNet(width, l) for l in lambdas]

def train_net(net: ConcatNet, dataloader):
    opt = torch.optim.AdamW(net.parameters())
    schedule = torch.optim.lr_scheduler.StepLR(opt, 10, 0.1)
    criterion = MeanRegularizedLoss(batch_size, net.lambda_wt)
    
    net.to(device).train()
    for ep in range(40):
        for (x_eb, x_hb, x_ho, y_true) in dataloader:
            x_eb, x_hb, x_ho = x_eb.float().to(device), x_hb.float().to(device), x_ho.float().to(device)
            y_true = y_true.float().to(device)

            opt.zero_grad()
            y_pred = net(x_eb, x_hb, x_ho)[:, 0]
            loss = criterion(y_pred, y_true)
            loss.backward()
            opt.step()

        print(f"Loss: {loss.item()}")
        schedule.step()
        
    net.eval()

for net in nets:
    train_net(net, dataloader)
# %%
tensor_data = get_normalized_hits(3.5, 15.)

def get_binned_predictions(net, tensor_data):
    x_eb1, x_hb1, x_ho1, y_true1 = tensor_data
    y_pred1 = net.to("cpu")(x_eb1.float(), x_hb1.float(), x_ho1.float()).detach()[:, 0]

    y_true = y_true1.exp()
    y_pred = y_pred1.exp()

    y_true2, args = torch.sort(y_true)
    y_pred2 = y_pred[args]
    lengs = [0]
    Es = [y_true2[0]]

    for (i,e) in enumerate(y_true2):
        if np.abs(e - Es[-1]) > 0.01:
            Es.append(e)
            lengs.append(i)
    lengs.append(len(y_true2))
    Es = np.asarray(Es)

    return y_pred2.numpy(), Es, np.asarray(lengs)

def get_binned_sums(tensor_data):
    x_eb1, x_hb1, _, y_true1 = tensor_data
    y_pred = (x_eb1.exp() - 1e-3).sum([1,2]) + (x_hb1.exp() - 1e-3).sum([1,2])
    y_true = y_true1.exp()

    y_true2, args = torch.sort(y_true)
    y_pred2 = y_pred[args]
    lengs = [0]
    Es = [y_true2[0]]

    for (i,e) in enumerate(y_true2):
        if np.abs(e - Es[-1]) > 0.01:
            Es.append(e)
            lengs.append(i)
    lengs.append(len(y_true2))
    Es = np.asarray(Es)

    return y_pred2.numpy(), Es, np.asarray(lengs)

def residual_analysis(y_pred, Es, lengs, n_histogram_bins=100):

    residual_modes = []
    residual_rms = []
    residual_means = []

    for (i,e) in enumerate(Es):
        residuals = e - y_pred[lengs[i]:lengs[i+1]]
        hist, bins = np.histogram(residuals, n_histogram_bins)
        bin_center = (bins[1:] + bins[:-1])/2
        mode = bin_center[np.argmax(hist)]
        residual_modes.append(mode)
        residual_means.append(np.mean(residuals))
        residual_rms.append(np.sqrt(np.mean(residuals**2)) / e)

    residual_rms = np.asarray(residual_rms)
    residual_modes = np.asarray(residual_modes)
    residual_means = np.asarray(residual_means)
    
    mean_E = (Es - residual_means) / Es
    mode_E = (Es - residual_modes) / Es
    
    print("Residual means: ", residual_means)
    print("Residual modes: ", residual_modes)
    print("Normalized means: ", mean_E)
    print("Normalized modes: ", mode_E)

    return residual_rms, mean_E, mode_E
# %% Compare cumulative and predicted stats
y_sum, Es, lengs = get_binned_sums(tensor_data)
y_pred, _, _ = get_binned_predictions(nets[0], tensor_data)

res_rms_sums, mean_E_sums, mode_E_sums = residual_analysis(y_sum, Es, lengs)
res_rms_pred, mean_E_pred, mode_E_pred = residual_analysis(y_pred, Es, lengs)
# %%
fig_rms, ax_rms = plt.subplots()
ax_rms.scatter(1/np.sqrt(Es), res_rms_sums, label="Summed hits")
ax_rms.scatter(1/np.sqrt(Es), res_rms_pred, label="Predicted energies")
ax_rms.set_xlabel('1 / sqrt(p)')
ax_rms.set_ylabel('RMS of Residuals')
ax_rms.set_title(f'Energy resolution at p')
ax_rms.legend()
ax_rms.grid(True)
plt.show(fig_rms)

fig_means, ax_means = plt.subplots()
ax_means.scatter(Es, mean_E_sums, label="Summed hits")
ax_means.scatter(Es, mean_E_pred, label="Predicted energies")
ax_means.axhline(1.0, color="black")
ax_means.set_xlabel('p')
ax_means.set_ylabel('<E> / p')
ax_means.grid(True)
ax_means.legend()
plt.show(fig_means)

fig_modes, ax_modes = plt.subplots()
ax_modes.scatter(Es, mode_E_sums, label="Summed hits")
ax_modes.scatter(Es, mode_E_pred, label="Predicted energies")
ax_modes.axhline(1.0, color="black")
ax_modes.set_xlabel('p')
ax_modes.set_ylabel('mode(y) / p')
ax_modes.grid(True)
ax_modes.legend()
plt.show(fig_means)
# %% Compare stats for different lambda's
res_rms_vals, mean_E_vals, mode_E_vals = [], [], []

fig_rms2, ax_rms2 = plt.subplots()
ax_rms2.set_xlabel('1 / sqrt(p)')
ax_rms2.set_ylabel('RMS of Residuals')
ax_rms2.set_title(f'Energy resolution at p')
ax_rms2.grid(True)

fig_means2, ax_means2 = plt.subplots()
ax_means2.axhline(1.0, color="black")
ax_means2.set_xlabel('p')
ax_means2.set_ylabel('<E> / p')
ax_means2.grid(True)

fig_modes2, ax_modes2 = plt.subplots()
ax_modes2.axhline(1.0, color="black")
ax_modes2.set_xlabel('p')
ax_modes2.set_ylabel('mode(E) / p')
ax_modes2.grid(True)

for net in nets:
    l = net.lambda_wt
    y_pred, _, _ = get_binned_predictions(net, tensor_data)
    res_rms, mean_E, mode_E = residual_analysis(y_pred, Es, lengs)

    ax_rms2.plot(1/np.sqrt(Es), res_rms, label=f"lambda = {l}")
    ax_means2.plot(Es, mean_E, label=f"lamda = {l}")
    ax_modes2.scatter(Es, mode_E, label=f"lamda = {l}")
ax_rms2.plot(1/np.sqrt(Es), res_rms_sums, label="Summed hits", color="black")

ax_rms2.legend()
ax_means2.legend()
ax_modes2.legend()

plt.show()
# %%
fig_rms.savefig("Figures/Summed_pred_rms.pdf")
fig_means.savefig("Figures/Summed_pred_means.pdf")
fig_modes.savefig("Figures/Summed_pred_modes.pdf")
fig_rms2.savefig("Figures/Lambda_rms.pdf")
fig_means2.savefig("Figures/Lambda_means.pdf")
fig_modes2.savefig("Figures/Lambda_modes.pdf")
# %%