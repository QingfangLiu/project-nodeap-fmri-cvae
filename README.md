# TMS Effect Analysis via Variational Autoencoders (VAEs)

This repository contains code and analyses for uncovering the effects of Transcranial Magnetic Stimulation (TMS) using resting-state fMRI data and Variational Autoencoders (VAEs). This is part of a cog neuro project with other parts of analysis hosted in another repo https://github.com/QingfangLiu/project-nodeap-core/tree/main

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
`data/SubConds.xlsx` provides subject-level information for model conditioning and analysis:

- **SubID**: Original subject code (e.g., NODEAP_*)  
- **StimLoc**: Stimulation site (Anterior vs. Posterior OFC)  
- **StimOrder**: Numeric code for session order  
- **tms_order_letters**: Human-readable sequence (C = cTBS, S = sham)  
- **Age**, **Sex**: Basic demographics  

  



