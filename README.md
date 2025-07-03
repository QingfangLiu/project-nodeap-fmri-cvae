# TMS Effect Analysis via Variational Autoencoders (VAEs)

This repository contains code and analyses for uncovering the effects of Transcranial Magnetic Stimulation (TMS) using resting-state fMRI data and Variational Autoencoders (VAEs).

This repo explores how a conditional VAE can learn meaningful representations from resting-state fMRI functional connectivity data.

We present two versions:

- **[v1_baseline](./v1_baseline/):** Initial implementation using subject-conditioned cVAE.
- **[v2_loso](./v2_loso/):** Extended version using Leave-One-Subject-Out cross-validation to test generalization.

