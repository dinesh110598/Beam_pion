# %%
import torch
from torch import nn
import numpy as np, matplotlib.pyplot as plt, h5py
from data_load import get_dataloader, get_normalized_hits
from scipy.stats import gaussian_kde
from concat_net import ConcatNet, MeanRegularizedLoss
# %%
batch_size = 128
width = 64
device = torch.device('cuda:0')
E0, E1, E2 = 0., 15., 500.
# %%
dataloader = get_dataloader(batch_size, min_E=E0, max_E=E1)
net = ConcatNet(width).to(device)
# %%
def train_net(net, dataloader):
    opt = torch.optim.AdamW(net.parameters())
    schedule = torch.optim.lr_scheduler.StepLR(opt, 3)
    criterion = MeanRegularizedLoss(batch_size, 1.0)
    
    net.to(device).train()
    for ep in range(14):
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
# %%
train_net(net, dataloader)
# %% Predict energies of all hits
x_eb1, x_hb1, x_ho1, y_true1 = get_normalized_hits(E0, E1)
# x_eb2, x_hb2, x_ho2, y_true2 = get_normalized_hits(E1, E2)

y_pred1 = net.to("cpu")(x_eb1.float(), x_hb1.float(), x_ho1.float()).detach()[:, 0]
# y_pred2 = net2.to("cpu")(x_eb2.float(), x_hb2.float(), x_ho2.float()).detach()[:, 0]

y_sum = (x_eb1.exp() - 1e-3).sum([1,2]) + (x_hb1.exp() - 1e-3).sum([1,2])
# y_pred = torch.cat([y_pred1, y_pred2])
# y_true = torch.cat([y_true1, y_true2])
y_true = y_true1.exp()
y_pred = y_pred1.exp()
# %% Bin all the same energy predictions
y_true2, args = torch.sort(y_true)
y_pred2 = y_pred[args]
y_sum2 = y_sum[args]
lengs = [0]
Es = [y_true2[0]]

for (i,e) in enumerate(y_true2):
    if np.abs(e - Es[-1]) > 0.01:
        Es.append(e)
        lengs.append(i)
lengs.append(len(y_true2))
Es = np.asarray(Es)
# %% Histogram predicted energies for every value of true energy
for (i,j) in zip(lengs[:-1], lengs[1:]):
    E = torch.round(torch.mean(y_true2[i:j].float())).int()
    plt.hist(y_pred2[i:j], 100)
    plt.axvline(E, color='black', linestyle='--')
    plt.title(f"E = {E} GeV")
    plt.show()
# %%
# Create a KDE object
kde = gaussian_kde(np.vstack([y_true, y_pred]))

# Generate the density grid
xi, yi = np.mgrid[y_true.min().item():y_true.max().item():100j, 
                  y_pred.min().item():y_pred.max().item():100j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
# %%
plt.scatter(y_true.exp(), y_pred.exp(), c=zi.reshape(xi.shape), cmap='viridis', alpha=0.5)
plt.colorbar(label='Density of points')
plt.plot(y_true.exp(), y_true.exp(), color="black")

plt.xscale('log')
plt.yscale('log')
plt.xlabel('True beam energy')
plt.ylabel('Predicted beam energy')
plt.grid(True)

plt.show()
# %% Eric's energy resolution code

def get_hist_mode(data, bins=100):
    counts, bin_edges = np.histogram(data, bins=bins)
    mode_bin_index = np.argmax(counts)
    mode_bin_start = bin_edges[mode_bin_index]
    mode_bin_end = bin_edges[mode_bin_index + 1]
    return (mode_bin_end + mode_bin_start) / 2


def compute_uncertainty(data, n_bootstraps=1000, bins=100):
    mode_bins = [
        get_hist_mode(np.random.choice(data, size=len(data), replace=True), bins=bins)
        for _ in range(n_bootstraps)
    ]
    return np.std(mode_bins)


def apply_correction(pred, bin_centers, residual_modes):
    coeff = np.polyfit(bin_centers + np.asarray(residual_modes), bin_centers, deg=2)
    correction_poly = np.poly1d(coeff)

    # plt.scatter(bin_centers + np.asarray(residual_modes), bin_centers, label="Histogram modes")
    # fit_domain = np.linspace(np.min(pred), np.max(pred), len(pred))
    # plt.plot(fit_domain, correction_poly(fit_domain), label='Predicted Value')
    # plt.xlabel('Predicted values')
    # plt.ylabel('Corrected values / True modes')
    # plt.legend()
    # plt.show()
    
    return correction_poly(pred), correction_poly


def calculate_binwise_statistics(true_values, predicted_values, bin_centers, n_bootstraps_error):
    # Create bin edges from bin centers
    bin_edges = np.concatenate([
        [bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
        (bin_centers[:-1] + bin_centers[1:]) / 2,
        [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2],
    ])
    
    residuals_std, mode_errors, residual_modes = [], [], []

    for i in range(len(bin_edges) - 1):
        bin_mask = (true_values >= bin_edges[i]) & (true_values < bin_edges[i + 1])
        residuals = predicted_values[bin_mask] - true_values[bin_mask]
        if len(residuals) > 0:
            residuals_std.append(np.std(residuals))
            residual_modes.append(get_hist_mode(residuals))
            mode_errors.append(compute_uncertainty(residuals, n_bootstraps=n_bootstraps_error))
        else:
            residuals_std.append(np.nan)
            residual_modes.append(np.nan)
            mode_errors.append(np.nan)

    return bin_centers, residuals_std, mode_errors, residual_modes


def run_analysis(true_values, predicted_values, model_name, bin_centers, lengs, n_bootstraps_mode, n_bootstraps_error):
    # Calculate statistics for uncorrected predictions
    bin_centers, residuals_std, mode_errors, residual_modes = calculate_binwise_statistics(
        true_values, predicted_values, bin_centers, n_bootstraps_error
    )

    # Apply correction and recalculate statistics
    corrected_preds, corr_poly = apply_correction(predicted_values, bin_centers, residual_modes)
    _, corrected_residuals_std, corrected_mode_errors, _ = calculate_binwise_statistics(
        true_values, corrected_preds, bin_centers, n_bootstraps_error
    )

    mean_E = []
    for (i,e) in enumerate(bin_centers):
        mean_E.append(np.mean(corrected_preds[lengs[i]:lengs[i+1]]) / e)

    plt.scatter(bin_centers, mean_E)
    plt.xlabel("p")
    plt.ylabel("<E>/p")
    plt.show()
        
    fig, ax = plt.subplots()
    # Plot uncorrected residuals
    ax.scatter(
        1/(bin_centers)**0.5, residuals_std / bin_centers, label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    ax.scatter(
        1/(bin_centers)**0.5,
        corrected_residuals_std / bin_centers,
        label='Corrected Residuals',
    )

    # Finalize the plot
    ax.set_xlabel('True Value')
    ax.set_ylabel('Standard Deviation of Residuals')
    ax.set_title(f'Residual Analysis: {model_name}')
    # ax.set_ylim(0., 0.5)
    ax.legend()
    ax.grid(True)
    plt.show(fig)

custom_bins = np.array([2, 3, 4, 5, 6, 7, 8, 9])  # Non-continuous bin edges
run_analysis(y_true_exp.numpy(), y_pred_exp.numpy(), model_name="MLP", bin_centers=Es, 
             lengs=lengs, n_bootstraps_mode=100, n_bootstraps_error=50)
run_analysis(y_true_exp.numpy(), y_sum2.numpy(), model_name="Cumulative", bin_centers=Es, 
             lengs=lengs, n_bootstraps_mode=100, n_bootstraps_error=50)
# %%
def get_hist_mode(data, bins=100):
    counts, bin_edges = np.histogram(data, bins=bins)
    mode_bin_index = np.argmax(counts)
    mode_bin_start = bin_edges[mode_bin_index]
    mode_bin_end = bin_edges[mode_bin_index + 1]
    return (mode_bin_end + mode_bin_start) / 2


def compute_uncertainty(data, n_bootstraps=1000, bins=100):
    mode_bins = [
        get_hist_mode(np.random.choice(data, size=len(data), replace=True), bins=bins)
        for _ in range(n_bootstraps)
    ]
    return np.std(mode_bins)


def apply_correction(pred, bin_centers, residuals_average):
    coeff = np.polyfit(bin_centers, residuals_average, deg=3)
    correction_poly = np.poly1d(coeff)
    residuals_average = np.array(residuals_average)
    plt.scatter(bin_centers, residuals_average, label='Residual Modes')
    fit_domain = np.linspace(bin_centers[0], bin_centers[-1], len(pred))
    plt.plot(fit_domain, correction_poly(fit_domain), label='Predicted Value')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual Mode')
    plt.legend()
    plt.show()
    return pred + correction_poly(pred)


def calculate_binwise_statistics(true_values, predicted_values, bin_centers, min_val, max_val, n_bootstraps_error):
    # Create bin edges from bin centers
    bin_edges = np.concatenate([
        [min_val],
        (bin_centers[:-1] + bin_centers[1:]) / 2,
        [max_val],
    ])

    residuals_std, mode_errors, residual_modes, residual_means, residuals_rms = [], [], [], [], []

    for i in range(len(bin_edges) - 1):
        bin_mask = (predicted_values >= bin_edges[i]) & (predicted_values < bin_edges[i + 1])
        residuals = true_values[bin_mask] - predicted_values[bin_mask]
        if len(residuals) > 0:
            residuals_std.append(np.std(residuals))
            residual_modes.append(get_hist_mode(residuals))
            mode_errors.append(compute_uncertainty(residuals, n_bootstraps=n_bootstraps_error))
            residual_means.append(np.mean(residuals))
            residuals_rms.append(np.sqrt(np.mean(residuals**2)))
        else:
            residuals_std.append(np.nan)
            residual_modes.append(np.nan)
            mode_errors.append(np.nan)
            residual_means.append(np.nan)
            residuals_rms.append(np.nan)

    return bin_centers, residuals_std, mode_errors, residual_modes, residual_means, residuals_rms


def run_analysis(true_values, predicted_values, model_name, bin_centers, min_val, max_val, n_bootstraps_mode, n_bootstraps_error):
    # Calculate statistics for uncorrected predictions
    bin_centers, residuals_std, mode_errors, residual_modes, residual_means, residual_rms = calculate_binwise_statistics(
        true_values, predicted_values, bin_centers, min_val, max_val, n_bootstraps_error
    )

    # Apply correction and recalculate statistics
    corrected_preds = apply_correction(predicted_values, bin_centers, residual_modes)
    bin_centers, corrected_residual_std, corrected_mode_errors, corrected_residual_modes, corrected_residual_means, corrected_residual_rms = calculate_binwise_statistics(
        true_values, corrected_preds, bin_centers, min_val, max_val, n_bootstraps_error
    )

    # Plot uncorrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers), residuals_std / bin_centers, label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers),
        corrected_residual_std / bin_centers,
        label='Corrected Residuals',
    )


    # Finalize the plot
    plt.xlabel('$1/\\sqrt{E}$')
    plt.ylabel('$rms/<E>$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot uncorrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers), residual_rms / bin_centers, label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers),
        corrected_residual_rms / bin_centers,
        label='Corrected Residuals',
    )


    # Finalize the plot
    plt.xlabel('$1/\\sqrt{E}$')
    plt.ylabel('$\\sigma/<E>$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot uncorrected residuals
    plt.scatter(
        bin_centers, residual_means, label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    plt.scatter(
        bin_centers,
        corrected_residual_means,
        label='Corrected Residuals',
    )


    # Finalize the plot
    plt.xlabel('$p_b$ (GeV/c)')
    plt.ylabel('<E>/$p_b$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot uncorrected residuals
    plt.scatter(
        bin_centers, residual_modes, label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    plt.scatter(
        bin_centers,
        corrected_residual_modes / bin_centers,
        label='Corrected Residuals',
    )


    # Finalize the plot
    plt.xlabel('$p_b$ (GeV/c)')
    plt.ylabel('mode(E)/$p_b$')
    plt.legend()
    plt.grid(True)
    plt.show()

plt.rcParams.update({'font.size': 16})

predicted_values = y_pred[lengs[2]:].numpy()
true_values = y_true[lengs[2]:].numpy()
bin_centers = np.asarray([4, 5, 6, 7, 8, 9])
run_analysis(true_values, predicted_values, model_name="CNN", bin_centers=bin_centers, min_val=min(predicted_values), max_val=max(predicted_values), n_bootstraps_mode=10000, n_bootstraps_error=10)
# %% My own implementation
def bin_predictions(y_pred, y_true):
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

    return y_pred2, Es, np.asarray(lengs)

def apply_correction(y_pred, Es, residual_centers):
    coeff = np.polyfit(Es - residual_centers, Es, 1)
    correction_fn = np.poly1d(coeff)
    y_pred_corr = correction_fn(y_pred)

    # plt.scatter(Es-residual_centers, Es)
    # line_range = np.linspace(np.min(y_pred), np.max(y_pred), len(y_pred))
    # plt.plot(line_range, correction_fn(line_range))
    # plt.grid(True)
    # plt.show()

    return y_pred_corr, correction_fn

def residual_analysis(y_pred, Es, lengs, n_histogram_bins, model_name="MLP"):

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
    y_pred_corr, _ = apply_correction(y_pred, Es, residual_means)
    
    residual_rms_corrected = []
    residual_modes_corrected = []
    residual_means_corrected = []
    for (i,e) in enumerate(Es):
        residuals = e - y_pred_corr[lengs[i]:lengs[i+1]]
        hist, bins = np.histogram(residuals, n_histogram_bins)
        bin_center = (bins[1:] + bins[:-1])/2
        mode = bin_center[np.argmax(hist)]
        mean = np.mean(residuals)
        residual_modes_corrected.append(mode)
        residual_means_corrected.append(mean)
        residual_rms_corrected.append(np.sqrt(np.mean(residuals**2)) / e)

    residual_rms_corrected = np.asarray(residual_rms_corrected)
    residual_modes_corrected = np.asarray(residual_modes_corrected)
    
    mean_E = (Es - residual_means) / Es
    mode_E = (Es - residual_modes) / Es
    mean_E_corrected = (Es - residual_means_corrected) / Es
    mode_E_corrected = (Es - residual_modes_corrected) / Es
    
    print("Residual means: ", residual_means)
    print("Residual modes: ", residual_modes)
    print("Normalized means: ", mean_E)
    print("Normalized modes: ", mode_E)

    plt.scatter(Es, mean_E, label="Means")
    plt.scatter(Es, mode_E, label="Modes")
    # plt.scatter(Es, mean_E_corrected, label="Means corrected")
    # plt.scatter(Es, mode_E_corrected, label="Modes corrected")
    plt.axhline(1.0, color="black")
    # plt.scatter(Es, residual_modes_corrected)
    plt.xlabel('p')
    plt.ylabel('<E> / p')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Residual RMS: ", residual_rms)
    fig, ax = plt.subplots()
    # Plot uncorrected residuals
    ax.scatter(
        1/Es**0.5, residual_rms, label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    # ax.scatter(
    #     1/Es**0.5, residual_rms_corrected, label='Corrected Residuals',
    # )

    # Finalize the plot
    ax.set_xlabel('1 / sqrt(p)')
    ax.set_ylabel('RMS of Residuals')
    ax.set_title(f'Energy resolution at p')
    # ax.set_ylim(0., 0.5)
    ax.legend()
    ax.grid(True)
    plt.show(fig)

y_pred2, Es, lengs = bin_predictions(y_pred, y_true)
y_pred2 = y_pred2[lengs[2]:]
Es, lengs = Es[2:], lengs[2:] - lengs[2]
assert len(y_pred2) == lengs[-1]
residual_analysis(y_pred2.numpy(), Es, lengs, 50)

# y_sum2, Es, lengs = bin_predictions(y_sum, y_true)
# residual_analysis(y_sum2.numpy(), Es, lengs, 100)
# %%
i = 1
y_true2[lengs[i]:lengs[i+1]].mean()
# %%
