<p align="center">
  <img src="media/logo-V2.jpeg" alt="EchoStrike Logo" width="400">
</p>

# EchoStrike: Acoustic Side-Channel Analysis on Keyboards 🎙️⌨️

![MATLAB](https://img.shields.io/badge/Language-MATLAB-blue.svg)
![Machine Learning](https://img.shields.io/badge/Topic-Machine%20Learning-orange.svg)
![Signal Processing](https://img.shields.io/badge/Topic-Signal%20Processing-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-blue.svg)
## Overview
<p align="center">
  <img src="media/result%20visualisation.png" alt="Topographic AI Prediction Map" width="600">
</p>

This repository contains a complete pipeline for analyzing, extracting, and deciphering acoustic side-channel emanations from physical keyboards. 

The primary objective of this project is to demonstrate how the unique sound profile of mechanical variations in plastic switches can be exploited to classify keystrokes using Machine Learning. This project scales from simple binary classification entirely up to a **full 104-class keyboard classification architecture** (Achieving ~71% accuracy on raw acoustic data alone).

> **Disclaimer:** This project was realized by Oussama AK-HAIL at Abdelmalek Essaâdi University (Morocco) strictly for educational signal analysis research. Its goal is to raise awareness regarding physical emission vulnerabilities and highlight protection protocols against acoustic eavesdropping.

---

## 🚀 The Full Keyboard Pipeline

To replicate the results or run the classification system on your own customized audio data, follow the MATLAB scripts in this exact order:

### 1. Automated Data Segmentation (`batch_segmentation.m`)
Handling raw audio files involving thousands of keystrokes is near-impossible manually. 
This script recursively scans directories containing raw audio recordings of typing, actively isolates transient acoustic peaks (keystrikes), and automatically crops the audio into uniform slices around the exact moment of impact. The isolated clicks are then saved into categorized subfolders. 

### 2. Feature Extraction & Augmentation (`prepare_full_keyboard_augmented.m`)
This is the core signal processing engine. It imports the segmented audio clips and constructs massive multi-dimensional datasets for the AI models.
- **Signal Condition:** Averages audio to mono.
- **Data Augmentation:** Mathematically clones the original clip via **Additive White Gaussian Noise (AWGN)** and **Pitch Shifting** to artificially triple the dataset to 30,000+ files, hardening the model against ambient background interference.
- **The Golden Pipeline:** The raw waveform undergoes Fast Fourier Transform (FFT) analysis. The positive spectrum is extracted, converted logarithmically (dB scale), strictly Min-Max Normalized, and downsampled into exactly 1000 spectral bins. 

### 3. Artificial Intelligence Training
To handle the complexity of predicting across all the different keys on a keyboard simultaneously, we utilize an advanced Support Vector Machine algorithm:

- **`train_full_keyboard.m` (Support Vector Machine):**
  Utilizes MATLAB's `fitcecoc` function to deploy a One-vs-All mechanism with an RBF Gaussian Kernel. Instead of just guessing between two keys, it natively trains distinct mathematical boundaries to differentiate across the entire keyboard layout.

<p align="center">
  <img src="media/3D%20PCA.png" alt="3D PCA Feature Space" height="350">
  <img src="media/full%20confusion%20matrix.png" alt="Full 104-Class Confusion Matrix" height="350">
</p>

### 4. Interactive Live Prediction (`predict_full_keyboard.m`)
The culmination of the project is the visualization engine. Loading the completed AI models, you can feed a blind audio sample into the script. The engine will:
1. Print the direct categorical prediction. 
2. Open a custom visual interface drawing a **2D Neon Topographic Heatmap**, charting the AI's complex probability densities directly atop a graphical, true-scale minimalist keyboard overlay. 
<p align="center">
  <img src="media/result%20visualisation.png" alt="Topographic AI Prediction Map" width="600">
</p>
---

## 🛡️ Countermeasures
Within our full PDF report (found in this repository), you can refer to Chapters 6 \& 7 which detail mathematical and physical defense measures designed to protect infrastructure from remote acoustic eavesdropping. These countermeasures involve mathematical spectral dampening algorithms operating beneath 1500Hz, and structural O-ring masking mechanics.

## Requirements
- MATLAB R2023a or higher.
- Signal Processing Toolbox installed.
- Statistics and Machine Learning Toolbox installed.

## Credits & Dataset License
*   **Author:** Oussama AK-HAIL 
*   **Dataset Source:** Raw audio utilized within this workflow originates from the [Multi-Keyboard Acoustic (MKA) Dataset](https://data.mendeley.com/datasets/bpt2hvf8n3/4), released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.
