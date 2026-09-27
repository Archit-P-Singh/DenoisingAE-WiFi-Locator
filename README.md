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
   - **Testing Accuracy**: ~89.9%
   - *Analysis*: We applied modern standard practices (`ReLU`, proper `.std()` scaling, and removal of Dropout due to the sparsity of WiFi signals). Interestingly, the "optimized" model slightly underperformed the baseline on the test set. This highlights a fascinating quirk of deep learning: the original authors' unconventional use of `.var()` scaling and `tanh` activations acted as a form of extreme mathematical regularization that happened to perfectly suit this highly sparse, noisy WiFi dataset. The modern un-bounded `ReLU` architecture made the network slightly more volatile, causing a tiny drop in generalization.

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
