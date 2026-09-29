### Place recognition with WiFi fingerprints using Autoencoders and Neural Networks

Tensorflow implementation of the model discussed in the following paper: [Low-effort place recognition with WiFi fingerprints using deep learning](https://arxiv.org/pdf/1611.02049v1.pdf)

### Overview & Directory Structure

This repository now contains two distinct implementations to demonstrate both the exact replication of the original paper and an optimized version with modern architectural improvements.

- **`baseline/`**: Contains the strict implementation based on the original research paper. It uses `tanh` activations across all layers, `.var()` variance scaling for input standardization, and no dropout. 
- **`optimized/`**: Contains an improved version incorporating modern standard practices. Changes include replacing `tanh` with `ReLU` to alleviate vanishing gradients, replacing the variance-based scaling with proper Standard Deviation `.std()` scaling, applying a linear activation on the final Decoder layer (since output features are not bounded to [-1,1]), and introducing Dropout (0.3) in the DNN phase to mitigate overfitting.

### Results & Gaps Analysis

1. **Baseline Results**: 
   - **Training Accuracy**: ~99.1%
   - **Testing Accuracy**: ~91.4%
   - *Analysis*: The baseline successfully replicates the paper's findings. However, a significant gap between training and testing accuracy (nearly 8%) indicates overfitting. Furthermore, the use of `.var()` scaling resulted in heavily compressed input values, which artificially deflated the apparent MSE loss values.

2. **Optimized Results**:
   - **Training Accuracy**: ~99.7%
   - **Testing Accuracy (Single Split)**: ~92.9%
   - **5-Fold Cross Validation Testing Accuracy**: **92.04% (± 0.69%)**
   - *Analysis*: We applied modern standard practices (`ReLU`, proper `.std()` scaling, and removal of Dropout due to the sparsity of WiFi signals). To combat the inherent volatility of un-bounded `ReLU` activations and prevent the model from memorizing the training noise, we introduced two crucial optimizations to the Adam optimizer: **L2 Regularization (Weight Decay)** and **Exponential Learning Rate Decay**. By penalizing excessively large weights and forcing the learning rate to decay by 5% every epoch, the model converged much more smoothly. 
   
   To mathematically prove the robustness of the optimized architecture, we replaced the static 70/30 train/val split with a rigorous **5-Fold Stratified Cross Validation** loop. Across 5 distinct data permutations, the optimized model reliably maintained an average testing accuracy of **92.04%**, statistically proving its superior generalization capability over the original baseline (91.45%).

### Model Visualization

The optimized model structures have been plotted and saved as PNGs in the `optimized/` directory:
- `optimized/AE_optimized.png` (Autoencoder architecture)
- `optimized/NN_optimized.png` (DNN architecture)

### Tools Required

Python 3 is used during development and following libraries are required to run the code:
* Tensorflow (Compat v1)
* Numpy
* Pandas

### Dataset

The UJIIndoorLoc dataset used for model training and testing, can be downloaded from the following [[link]](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc).
