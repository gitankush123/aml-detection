# Dual-Branch HDC-VAE for Anomaly Detection (AML)

This repository contains the implementation and pre-trained weights for the **Dual-Branch Hyperdimensional Computing (HDC) Variational Autoencoder (VAE)** anomaly detection architecture.

## Architecture Overview

The system processes input images through two parallel pathways:
* **Global Branch:** Maps the VAE posterior mean ($\mathbf{z}_{\text{mean}} \in \mathbb{R}^{64}$) into a 10,000-dimensional hypervector using fixed random projection.
* **Local Branch:** Computes localized reconstruction error heatmaps, extracts patch-level features across an $8\times8$ grid, and projects them into spatial hypervectors.
* **HDC Fusion:** Combines global latent state representation and localized reconstruction error hypervectors for final similarity scoring against a trained normal prototype.

---

## Repository Structure

```text
├── app.py              # Streamlit deployment app
├── proto_normal.npy    # Pre-computed normal prototype hypervector
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
