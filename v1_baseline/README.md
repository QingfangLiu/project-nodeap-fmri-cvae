# Goal of v1_baseline

To identify and characterize the effect of TMS on brain connectivity patterns by applying VAEs and Conditional VAEs (cVAEs) to resting-state fMRI-derived functional connectivity (FC) data.


## Model Variants

1. `VAE` on AAL-based FC as a sanity check on subject identification using VAEs
2. `VAE` on AAL-sphere FC  
3. `Conditional VAE (cVAE)` on AAL-sphere FC, with subject ID added to both encoder and decoder


- **Versions:**
  - Raw FC
  - Weighted FC (with/without resampling to handle class imbalance)


## Analyses logic

- Session-wise latent space distances to the **null (control)** condition
- assumption: the distance (in latent space) between null and sham should be smaller than distance from null and cTBS







