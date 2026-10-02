#!/usr/bin/env python3
"""Posterior EEG spectral analysis for the CLN3 study.

Loads epoched EEGLAB ``.set`` files, computes Welch power spectra,
parameterizes spectra with FOOOF, and extracts posterior spectral measures
used in the manuscript.
"""

import mne
import numpy as np
from fooof import FOOOF
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import find_peaks
import pandas as pd
import sys

def fit_fooof(epochs, file_code, subject_id_for_plot, t_value_for_plot,
              freq_range=[1, 35], peaks_threshold=2, picks='eeg', plot_results=True):
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
        periodic_lin = (10**model_log10) - (10**ap_log10)   # <—
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
            print(f"  Info: No local maxima 5–13 Hz for {file_code}")
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

# ========================= MAIN =========================
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    # --- Step 1: Select the Subject Information File (CSV/Excel) ---
    subject_info_file_path = filedialog.askopenfilename(
        title="Select the Subject Information File (CSV or Excel)",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")]
    )
    if not subject_info_file_path:
        print("No subject information file selected. Exiting.")
        exit()

    subject_info_df = pd.DataFrame()
    selected_sheet_name = ""

    try:
        if subject_info_file_path.endswith('.csv'):
            subject_info_df = pd.read_csv(subject_info_file_path)
            selected_sheet_name = os.path.splitext(os.path.basename(subject_info_file_path))[0]
        elif subject_info_file_path.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(subject_info_file_path)
            sheet_names = xls.sheet_names
            if not sheet_names:
                messagebox.showerror("Error", "No sheets found in the Excel file.")
                exit()

            sheet_selection_window = tk.Toplevel(root)
            sheet_selection_window.title("Select Sheet")
            tk.Label(sheet_selection_window, text="Select the sheet containing 'Code', 'Subject', 'T', 'Age':").pack(pady=10)
            selected_sheet_var = tk.StringVar(sheet_selection_window)
            selected_sheet_var.set(sheet_names[0])
            dropdown = tk.OptionMenu(sheet_selection_window, selected_sheet_var, *sheet_names)
            dropdown.pack(pady=5)

            def confirm_selection():
                global selected_sheet_name
                selected_sheet_name = selected_sheet_var.get()
                sheet_selection_window.destroy()

            tk.Button(sheet_selection_window, text="Confirm", command=confirm_selection).pack(pady=10)
            sheet_selection_window.update_idletasks()
            x = root.winfo_x() + (root.winfo_width() // 2) - (sheet_selection_window.winfo_width() // 2)
            y = root.winfo_y() + (root.winfo_height() // 2) - (sheet_selection_window.winfo_height() // 2)
            sheet_selection_window.geometry(f"+{x}+{y}")
            root.wait_window(sheet_selection_window)

            if selected_sheet_name:
                subject_info_df = pd.read_excel(subject_info_file_path, sheet_name=selected_sheet_name)
            else:
                messagebox.showerror("Error", "No sheet selected. Exiting.")
                exit()
        else:
            print("Unsupported file format for subject information. Please select a CSV or Excel file.")
            exit()
    except Exception as e:
        print(f"Error loading subject information file: {e}")
        sys.exit()

    if 'Code' not in subject_info_df.columns:
        messagebox.showerror("Column Error", "The selected file/sheet does not contain a column named 'Code'. Please ensure the header is 'Code'.")
        print("Error: 'Code' column not found in the loaded subject information. Please check your file header.")
        exit()

    expected_cols = ['Subject', 'T', 'Age']
    for col in expected_cols:
        if col not in subject_info_df.columns:
            print(f"Warning: Column '{col}' not found...")

    print("\n--- Loaded Subject Information Table ---")
    print(subject_info_df.head())
    print(f"Total entries in subject info: {len(subject_info_df)}\n")

    # --- Step 2: Select EEG folder ---
    eeg_data_parent_folder = filedialog.askdirectory(title="Select the parent folder containing all EEG subfolders (e.g., 'ProcessedEEGs/CLN3_All')")
    if not eeg_data_parent_folder:
        print("No EEG data parent folder selected. Exiting.")
        exit()

    print(f"EEG data will be searched in subfolders within: {eeg_data_parent_folder}\n")

    results_base_folder = os.path.dirname(subject_info_file_path) if subject_info_file_path else eeg_data_parent_folder
    results_folder = os.path.join(results_base_folder, "_Results_EEG_Analysis")
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results (plots, final table) will be saved in: {results_folder}")

    posterior_channels = ['P3', 'P4', 'O1', 'O2']

    all_results_list = []
    processed_eeg_codes = []

    # --- Step 3: Iterate ---
    for index, row in subject_info_df.iterrows():
        file_code = row['Code']
        subject_from_table = row['Subject'] if 'Subject' in row else 'N/A'
        t_value = row['T'] if 'T' in row else 'N/A'
        age = row['Age'] if 'Age' in row else np.nan

        subject_subfolder = f"{file_code}_subfold"
        eeg_filename = f"{file_code}_PreProc.set"
        set_file_path = os.path.join(eeg_data_parent_folder, subject_subfolder, eeg_filename)

        if os.path.exists(set_file_path):
            print(f"Processing: Code='{file_code}', Subject='{subject_from_table}', T='{t_value}', Age={age}")
            print(f"  Attempting to load: {set_file_path}")
            try:
                epochs = mne.read_epochs_eeglab(set_file_path, verbose=False)
                fooof_results, fig = fit_fooof(
                    epochs, file_code, subject_from_table, t_value,
                    picks=posterior_channels, plot_results=True, freq_range=[3, 35]
                )

                if fooof_results:
                    subject_data = {
                        'Code': file_code,
                        'Subject': subject_from_table,
                        'T': t_value,
                        'Age': age,
                        'Exponent': fooof_results['exponent'],
                        'Offset': fooof_results['offset'],
                        'RSquared': fooof_results['r_squared'],
                        'ThetaPower': fooof_results['theta_power'],
                        'AlphaPower': fooof_results['alpha_power'],
                        'BetaPower': fooof_results['beta_power'],
                        'IndividualPeakFreq': fooof_results['individual_peak_freq'],
                        'IndividualPeakPower': fooof_results['individual_peak_power'],
                        'AlphaThetaRatio': fooof_results['alpha_theta_ratio'],
                    }
                    all_results_list.append(subject_data)
                    processed_eeg_codes.append(file_code)

                    if fig:
                        plot_filename = os.path.join(results_folder, f"{file_code}_FOOOF_Posterior_Analysis.png")
                        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  Plot saved to {plot_filename}")

                else:
                    print(f"  FOOOF analysis failed for {file_code}.")
            except Exception as e:
                print(f"  Error processing {file_code}: {e}")
        else:
            print(f"  Skipping: EEG file not found for Code='{file_code}' at expected path: '{set_file_path}'")

    # --- Step 4: Save final table ---
    if all_results_list:
        final_table_to_save = pd.DataFrame(all_results_list)
        print("\n--- Final Table with Subject Info and Computed EEG Measures ---")
        print(final_table_to_save.to_string())

        output_csv_filename = os.path.join(results_folder, f"{selected_sheet_name}_complete_posterior_analysis.csv")
        try:
            final_table_to_save.to_csv(output_csv_filename, index=False)
            print(f"\nAll FOOOF results (including subject info) saved to {output_csv_filename}")
        except Exception as e:
            print(f"Error saving all FOOOF results to CSV: {e}")
    else:
        print("No subjects were processed or no results were obtained. Check your subject information file and EEG data folder.")

    root.destroy()
