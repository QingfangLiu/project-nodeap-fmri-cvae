# TMS Effect Analysis via Variational Autoencoders (VAEs)

This repository contains standalone code and analyses for examining the effects of Transcranial Magnetic Stimulation (TMS) on brain connectivity using resting-state fMRI and Variational Autoencoders (VAEs).  
It is part of a larger cognitive neuroscience project; other components of the analysis can be found in the companion repository: [project-nodeap-core](https://github.com/QingfangLiu/project-nodeap-core/tree/main).

---

### Background  
Transcranial Magnetic Stimulation (TMS) is a widely used non-invasive technique in cognitive neuroscience for probing the causal role of brain regions in cognition.  
Here, we investigate whether deep learning approaches can detect subtle TMS-induced changes in brain connectivity.

### Data  
We focus on functional connectivity (FC) derived from resting-state fMRI, collected after participants received either sham stimulation or continuous theta burst stimulation (cTBS).

### Research Question  
The central question is straightforward:  
*Can resting-state FC reveal neural evidence of TMS effects?*

### Motivation  
Although non-invasive stimulation often produces modest and difficult-to-detect neural changes, TMS in our study did alter behavioral task performance.  
This motivates the search for converging neural evidence to better understand the origins of the observed behavioral effects.


There are two versions:

- **[v1_baseline](./v1_baseline/):** Initial implementation using VAE and subject-conditioned cVAE fitted to all subjects. 
- **[v2_loso](./v2_loso/):** Extended version using Leave-One-Subject-Out cross-validation to circumvent model overfitting. 


---

## Repository Structure

```text
├── data/
│   ├── data_FC/                    # FC matrices for each subject/session
│   │   └── NODEAP_06/
│   │       └── D0/
│   │           ├── conn_matrix.mat
│   │           └── conn_matrix_w_sphere.mat
│   └── subject_info.xlsx          # Metadata for all subjects
│
├── v1_baseline/                   # Scripts and configs for the baseline model
├── v2_loso/                       # Scripts and configs for LOSO evaluation
├── utils/                         # Shared helper functions
└── README.md                      # You are here
```

---

## 📂 Data Overview

- **Functional Connectivity Matrices**  
  Located in `data/data_FC/`, organized by subject and session. Each session folder includes:
  - `conn_matrix.mat`: AAL-based FC matrix
  - `conn_matrix_w_sphere.mat`: AAL-based FC matrix between AAL ROIs and spherical ROIs

**Subject Metadata**  
`data/SubConds.xlsx` provides subject-level information for model conditioning and analysis (copied from the core repo):

- **SubID**: Original subject code (e.g., NODEAP_*)  
- **StimLoc**: Stimulation site (Anterior vs. Posterior OFC)  
- **StimOrder**: Numeric code for session order  
- **tms_order_letters**: Human-readable sequence (C = cTBS, S = sham)  
- **Age**, **Sex**: Basic demographics  

  



