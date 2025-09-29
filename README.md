# TMS Effect Analysis via Variational Autoencoders (VAEs)

This repository contains standalone code and analyses for examining the effects of Transcranial Magnetic Stimulation (TMS) on brain connectivity using resting-state fMRI and Variational Autoencoders (VAEs).  
It is part of a larger cognitive neuroscience project; other components of the analysis can be found in the companion repository: [project-nodeap-core](https://github.com/QingfangLiu/project-nodeap-core/tree/main).

The resting-state fMRI data have been publicly deposited on [OpenNeuro (ds006693)](https://openneuro.org/datasets/ds006693).

---

### Background  
Transcranial Magnetic Stimulation (TMS) is a widely used non-invasive technique in cognitive neuroscience for probing the causal role of brain regions in cognition. Here, we investigate whether deep learning approaches can detect subtle TMS-induced changes in brain connectivity.

Although non-invasive stimulation often produces modest and difficult-to-detect neural changes, TMS in our study did alter behavioral task performance. This motivates the search for converging neural evidence to better understand the origins of the observed behavioral effects.

### Research Question  
The central question is straightforward:  
*Can resting-state FC reveal neural evidence of TMS effects?*

### Data  
We focus on functional connectivity (FC) derived from resting-state fMRI, collected after participants received either sham stimulation or continuous theta burst stimulation (cTBS).

### Proposed Solution

I proposed using unsupervised deep learning methods—specifically a Variational Autoencoder (VAE) and its conditional variant (CVAE)—because of their ability to extract latent embedding dimensions. My working hypothesis was that cTBS would produce a greater shift in the embedded vectors than sham stimulation, relative to the null (baseline) session collected on the initial experiment day.

## Model Variants  

I tested several VAE-based models with different input features and training schemes:  

1. **VAE on AAL-based FC**  
   - Sanity check for subject identification using VAEs.  

2. **VAE on AAL-sphere FC**  
   - Restricts analysis to seed and target ROIs.  
   - Involves **4 ROIs**:  
     - `0`: aOFC seed  
     - `1`: aOFC target  
     - `2`: pOFC seed  
     - `3`: pOFC target  
   - See [fc_AAL_spheres scripts](https://github.com/QingfangLiu/project-nodeap-core/tree/main/scripts/fc_AAL_spheres) for implementation details.  

3. **Conditional VAE (cVAE) on AAL-sphere FC**  
   - Adds subject ID as a condition to both encoder and decoder.  

### Versions  
- **Raw** (`raw`): Unmodified data.  
- **Weighted** (`wtd`): Resampled to address class imbalance.  


### Findings

cTBS over the anterior OFC (aOFC) produced greater deviations in functional connectivity from baseline compared to sham stimulation, with no effects for other ROIs. Moreover, in the aOFC group, connectivity changes predicted individual differences in subsequent behavior, linking anterior (but not posterior) OFC stimulation to learning of specific stimulus–outcome associations.

For more details on the analyses, equations, behavioral links, and plots, see our manuscript:  

Liu et al., *Distinct contributions of anterior and posterior orbitofrontal cortex to outcome-guided behavior*,  
*Current Biology* (in press). 


---

## Repository Structure

```text
project-nodeap-fmri-cvae/
├── data/                          
│   ├── data_FC/                   # Functional connectivity (FC) matrices per subject/session
│   │   └── NODEAP_06/             # Example subject folder
│   │       └── D0/                # Example session folder (Day 0 / baseline)
│   │           ├── conn_matrix.mat
│   │           └── conn_matrix_w_sphere.mat
│   └── SubConds.xlsx              # Subject condition metadata
│
├── outputs/                       # Model outputs
│
├── scripts/                       # Main analysis scripts
│   ├── vae/                       # VAE model notebooks
│   │   ├── vae_fc_aal_raw.ipynb
│   │   ├── vae_fc_aal_wtd.ipynb
│   │   ├── vae_fc_aal_spheres_raw.ipynb
│   │   └── vae_fc_aal_spheres_wtd.ipynb
│   │
│   ├── cvae/                      # Conditional VAE model notebooks
│   │   ├── cvae_fc_aal_spheres_raw.ipynb
│   │   └── cvae_fc_aal_spheres_wtd.ipynb
│   │
│   └── utils/                     # Shared helper modules
│       ├── bootstrap.py           # Bootstrapping procedures
│       ├── data_utils.py          # Data loading and preprocessing
│       ├── models.py              # Model architectures (VAE / CVAE)
│       ├── train_utils.py         # Training loops and evaluation
│       └── utils_plotting.py      # Plotting and visualization utilities
│
└── README.md                      # Main project description
```

---

## Data Overview  

- **Functional Connectivity (FC) Matrices**  
  Stored in `data/data_FC/`, organized by subject and session. Each session folder contains:  
  - `conn_matrix.mat`: AAL-based FC matrix  
  - `conn_matrix_w_sphere.mat`: FC matrix between AAL ROIs and spherical ROIs (seed/target focused)  

- **Subject Metadata**  
  `data/SubConds.xlsx` contains subject-level information used for model conditioning and analysis (copied from the core repo):  
  - **SubID**: Subject identifier (e.g., `NODEAP_*`)  
  - **StimLoc**: Stimulation site (Anterior vs. Posterior OFC)  
  - **StimOrder**: Numeric session order code  
  - **tms_order_letters**: Human-readable session sequence (C = cTBS, S = sham)  
  - **Age**, **Sex**: Demographic information  



