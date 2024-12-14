import numpy as np
import matplotlib.pyplot as plt

def get_hist_mode(data, bins=100, bin_shift=0.0):
    data_min, data_max = np.min(data), np.max(data)
    bin_width = (data_max - data_min) / bins
    start = data_min + bin_shift * bin_width
    end = data_max + bin_shift * bin_width
    bin_edges = np.linspace(start, end, bins + 1)

    counts, _ = np.histogram(data, bins=bin_edges)
    mode_bin_index = np.argmax(counts)
    mode_bin_start = bin_edges[mode_bin_index]
    mode_bin_end = bin_edges[mode_bin_index + 1]

    return 0.5 * (mode_bin_start + mode_bin_end)


def compute_uncertainty(data, n_bootstraps=1000, bins=100):
    mode_estimates = []
    n = len(data)
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bin_shift = np.random.uniform(-0.5, 0.5)
        mode_estimate = get_hist_mode(sample, bins=bins, bin_shift=bin_shift)
        mode_estimates.append(mode_estimate)

    return np.std(mode_estimates)


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


def calculate_binwise_statistics(true_values, predicted_values, bin_centers, n_bootstraps_error):
    # Create bin edges from bin centers
    bin_edges = np.concatenate([
        bin_centers[0:1],
        (bin_centers[:-1] + bin_centers[1:]) / 2,
        bin_centers[-1:]
    ])
    print(bin_edges)

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


def run_analysis(true_values, predicted_values, model_name, bin_centers, n_bootstraps_mode, n_bootstraps_error):
    # Calculate statistics for uncorrected predictions
    bin_centers, residuals_std, mode_errors, residual_modes, residual_means, residual_rms = calculate_binwise_statistics(
        true_values, predicted_values, bin_centers, n_bootstraps_error
    )

    # Apply correction and recalculate statistics
    corrected_predictions = apply_correction(predicted_values, bin_centers, residual_modes)

    mean_preds = []
    rms_preds = []
    std_preds = []
    for val in np.unique(true_values):
        p = predicted_values[true_values == val]
        mean_preds.append(np.mean(p))
        rms_preds.append(np.sqrt(np.mean(p**2)))
        std_preds.append(np.std(p))
    mean_preds = np.array(mean_preds)
    rms_preds = np.array(rms_preds)
    std_preds = np.array(std_preds)
    print(len(mean_preds))
    print(len(std_preds))
    print(len(bin_centers))

    mean_preds_corrected = []
    rms_preds_corrected = []
    std_preds_corrected = []
    for val in np.unique(true_values):
        c = corrected_predictions[true_values == val]
        mean_preds_corrected.append(np.mean(c))
        rms_preds_corrected.append(np.sqrt(np.mean(c**2)))
        std_preds_corrected.append(np.std(c))
    mean_preds_corrected = np.array(mean_preds_corrected)
    rms_preds_corrected = np.array(rms_preds_corrected)
    std_preds_corrected = np.array(std_preds_corrected)

    # Plot uncorrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers),
        std_preds / mean_preds,
        label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers),
        std_preds_corrected / mean_preds_corrected,
        label='Corrected Residuals',
    )


    # Finalize the plot
    plt.xlabel('$1/\\sqrt{p_{true}}$')
    plt.ylabel('$std(p_{pred})/<p_{pred}>$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot uncorrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers),
        rms_preds / mean_preds,
        label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    plt.scatter(
        1 / np.sqrt(bin_centers),
        rms_preds_corrected / mean_preds_corrected,
        label='Corrected Residuals',
    )

    # Finalize the plot
    plt.xlabel('$1/\\sqrt{p_{true}}$')
    plt.ylabel('$rms(p_{pred})/<p_{pred}>$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot uncorrected residuals
    plt.scatter(
        bin_centers, mean_preds / bin_centers, label='Uncorrected Residuals'
    )

    # Plot corrected residuals
    plt.scatter(
        bin_centers,
        mean_preds_corrected / bin_centers,
        label='Corrected Residuals',
    )


    # Finalize the plot
    plt.xlabel('$p_{true}$ (GeV/c)')
    plt.ylabel('$<p_{pred}>/p_{true}$')
    plt.legend()
    plt.grid(True)
    plt.show()