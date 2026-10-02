#!/usr/bin/env python3
"""Batch posterior EEG spectral analysis for epoched EEGLAB datasets.

This script applies the same Welch, FOOOF, band-power, and posterior-dominant
peak settings used in ``eeg_spectral_analysis.py``. It is provided as a
folder-based batch alternative when no subject-information table is needed.
"""

import os
import csv
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import mne
import numpy as np
from fooof import FOOOF
from scipy.signal import find_peaks

def fit_fooof(epochs, file_code, subject_id_for_plot, t_value_for_plot,
              freq_range=[3, 35], peaks_threshold=2, picks='eeg', plot_results=True):
    """
    Compute posterior spectral measures from epoched EEG data.

    The periodic component is reconstructed in linear power units as the
    full FOOOF model minus the fitted aperiodic component. Band powers and
    posterior dominant peak estimates are derived from this component.
    """

    sfreq = epochs.info['sfreq']
    n_per_seg = int(sfreq * 2)
    n_overlap = n_per_seg // 2

    # --- Power spectral density ---
    try:
        psds, freqs = epochs.compute_psd(
            method="welch",
            picks=picks,
            fmin=freq_range[0],
            fmax=freq_range[1],
            n_fft=n_per_seg,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            window="hann",
            verbose=False
        ).get_data(return_freqs=True)

        # Average across epochs and selected channels
        psd_avg = np.mean(psds, axis=(0, 1))

    except Exception as e:
        print(f"PSD computation error for {file_code}: {e}")
        return None, None

    # --- FOOOF model fitting ---
    fm = FOOOF(peak_width_limits=[0.5, 13.0], peak_threshold=peaks_threshold,
               aperiodic_mode='fixed', verbose=False)
    try:
        fm.fit(freqs, psd_avg, freq_range)
        # Reconstruct the periodic component in linear power units
        model_log10 = np.array(fm.get_model())
        ap_log10 = np.array(fm._ap_fit)
        periodic_lin = (10**model_log10) - (10**ap_log10)
    except Exception as e:
        print(f"FOOOF fitting error for {file_code}: {e}")
        return None, None

    # --- Band powers and alpha/theta ratio ---
    theta_power, alpha_power, beta_power = np.nan, np.nan, np.nan
    individual_peak_freq, individual_peak_power = np.nan, np.nan
    alpha_theta_ratio = np.nan

    if periodic_lin is not None and freqs is not None:
        bands = {'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30]}
        band_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
            if np.any(idx_band):
                band_powers[band_name] = np.trapz(periodic_lin[idx_band], freqs[idx_band])  # trapz
            else:
                band_powers[band_name] = np.nan

        theta_power = band_powers.get('theta', np.nan)
        alpha_power = band_powers.get('alpha', np.nan)
        beta_power  = band_powers.get('beta', np.nan)

        alpha_theta_ratio = (alpha_power / theta_power) if (np.isfinite(alpha_power) and np.isfinite(theta_power) and theta_power > 0) else np.nan

    # --- Posterior dominant peak detection ---
    search_min, search_max = 5.0, 12.0
    idx_band = (freqs >= search_min) & (freqs <= search_max)
    
    individual_peak_freq, individual_peak_power = np.nan, np.nan
    
    if np.any(idx_band):
        spec_band = periodic_lin[idx_band]
        freq_band = freqs[idx_band]
    
        # Normalize within the search band so prominence is scale-independent
        spec_norm = (spec_band - np.min(spec_band)) / (np.ptp(spec_band) + 1e-18)
    
        # Find local maxima in the search band
        peaks, props = find_peaks(spec_norm, prominence=0.05)
    
        if len(peaks) > 0:
            # Select the peak with the largest prominence
            pk = peaks[np.argmax(props["prominences"])]
            individual_peak_freq = float(freq_band[pk])
            individual_peak_power = float(spec_band[pk])
            print(f"  Info: Peak @ {individual_peak_freq:.2f} Hz for {file_code}")
        else:
            print(f"  Info: No local maxima 5–12 Hz for {file_code}")
    else:
        print(f"  Warning: No frequencies between {search_min}–{search_max} Hz for {file_code}")


    # --- Diagnostic plots ---
    fig = None
    if plot_results:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Plot title
        title_str = f"FOOOF Analysis - Subject: {subject_id_for_plot}, T: {t_value_for_plot}\nElectrodes: {', '.join(picks)}"
        fig.suptitle(title_str, fontsize=16)

        # Top Left: Original PSD (linear)
        axs[0, 0].plot(freqs, psd_avg, label='Original PSD')
        axs[0, 0].set_title('Original Power Spectral Density')
        axs[0, 0].set_xlabel('Frequency (Hz)')
        axs[0, 0].set_ylabel('Power Spectral Density')
        axs[0, 0].legend()

        # Top Right: FOOOF Fit (linear)
        axs[0, 1].plot(freqs, psd_avg, label='Original PSD')
        axs[0, 1].plot(freqs, 10**model_log10, '--', label='FOOOF Fit (linear)')
        axs[0, 1].set_title('FOOOF Model Fit (Linear Scale)')
        axs[0, 1].set_xlabel('Frequency (Hz)')
        axs[0, 1].set_ylabel('Power Spectral Density')
        axs[0, 1].legend()

        # Bottom Left: Aperiodic component (linear)
        axs[1, 0].plot(freqs, 10**ap_log10, '--', label='Aperiodic (linear)', color='orange')
        axs[1, 0].set_title('Aperiodic Component (Linear Scale)')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Power Spectral Density')
        axs[1, 0].legend()

        # Bottom Right: Periodic component (linear) + peak
        axs[1, 1].plot(freqs, periodic_lin, label='Periodic (Aperiodic-corrected)', color='green')
        if not np.isnan(individual_peak_freq):
            axs[1, 1].axvline(individual_peak_freq, color='red', linestyle=':', label=f'Identified Alpha Peak ({individual_peak_freq:.2f} Hz)')
            idx_peak = np.argmin(np.abs(freqs - individual_peak_freq))
            axs[1, 1].plot(individual_peak_freq, periodic_lin[idx_peak], 'ro', markersize=8)
        axs[1, 1].set_title('Periodic Component (Linear)')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Power Spectral Density (Periodic)')
        axs[1, 1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # --- Results ---
    results = {
        'exponent': fm.aperiodic_params_[1],
        'offset': fm.aperiodic_params_[0],
        'peak_params': fm.peak_params_.tolist(),
        'r_squared': fm.r_squared_,
        'error': fm.error_,
        # Corrected spectrum is not exported
        'frequencies': freqs.tolist(),
        'theta_power': theta_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'individual_peak_freq': individual_peak_freq,
        'individual_peak_power': individual_peak_power,
        'alpha_theta_ratio': alpha_theta_ratio
    }

    return results, fig

POSTERIOR_CHANNELS = ["P3", "P4", "O1", "O2"]
FREQ_RANGE = [3, 35]


def main():
    root = tk.Tk()
    root.withdraw()

    main_folder = filedialog.askdirectory(
        title="Select the folder containing EEG subject subfolders"
    )
    if not main_folder:
        print("No EEG folder selected. Exiting.")
        return

    output_folder = os.path.join(main_folder, "_Results_EEG_Analysis")
    os.makedirs(output_folder, exist_ok=True)

    rows = []
    subject_folders = sorted(
        entry.path for entry in os.scandir(main_folder) if entry.is_dir()
    )

    for subject_folder in subject_folders:
        folder_name = os.path.basename(subject_folder)
        subject_id = folder_name[:-8] if folder_name.endswith("_subfold") else folder_name

        set_files = sorted(
            f for f in os.listdir(subject_folder) if f.lower().endswith(".set")
        )
        if not set_files:
            continue

        set_file = os.path.join(subject_folder, set_files[0])
        print(f"Processing {subject_id}: {set_file}")

        try:
            epochs = mne.read_epochs_eeglab(set_file, verbose=False)
            results, fig = fit_fooof(
                epochs,
                file_code=subject_id,
                subject_id_for_plot=subject_id,
                t_value_for_plot="N/A",
                freq_range=FREQ_RANGE,
                picks=POSTERIOR_CHANNELS,
                plot_results=True,
            )
        except Exception as exc:
            print(f"  Error processing {subject_id}: {exc}")
            continue

        if results is None:
            continue

        rows.append({
            "SubjectID": subject_id,
            "Exponent": results["exponent"],
            "Offset": results["offset"],
            "RSquared": results["r_squared"],
            "ThetaPower": results["theta_power"],
            "AlphaPower": results["alpha_power"],
            "BetaPower": results["beta_power"],
            "IndividualPeakFreq": results["individual_peak_freq"],
            "IndividualPeakPower": results["individual_peak_power"],
            "AlphaThetaRatio": results["alpha_theta_ratio"],
        })

        if fig is not None:
            plot_path = os.path.join(
                output_folder, f"{subject_id}_FOOOF_Posterior_Analysis.png"
            )
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    if rows:
        output_csv = os.path.join(output_folder, "posterior_spectral_measures.csv")
        with open(output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to: {output_csv}")
    else:
        print("No EEG datasets were successfully processed.")

    root.destroy()


if __name__ == "__main__":
    main()
