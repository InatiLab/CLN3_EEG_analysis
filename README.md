# CLN3 EEG analysis

Analysis code for the study of quantitative EEG biomarkers in CLN3 disease.

The repository contains scripts for:

* preprocessing-derived EEG spectral analysis
* FOOOF parameterization of posterior EEG spectra
* posterior dominant peak detection
* theta, alpha, and beta power estimation
* alpha/theta ratio calculation
* normative modeling using age and sex

## Data

The preprocessed EEG dataset is publicly available on OpenNeuro:

**Dataset:** ds008896
**DOI:** 10.18112/openneuro.ds008896.v1.0.0

The dataset includes individuals with genetically confirmed CLN3 disease and NIH healthy volunteers.

## Scripts

`eeg_spectral_analysis.py`
Primary posterior EEG spectral analysis workflow.

`fooof_batch_analysis.py`
Batch spectral parameterization and extraction of EEG features.

`normative_modeling.py`
Normative modeling of EEG measures using the PCNtoolkit.

## Software

Core dependencies include:

* Python
* MNE-Python
* NumPy
* pandas
* SciPy
* matplotlib
* FOOOF
* PCNtoolkit

## Notes

Posterior EEG analyses focus on channels P3, P4, O1, and O2. For the reported spectral analyses, these channels were referenced to Fz before spectral estimation.

See the associated manuscript for complete methodological details.
