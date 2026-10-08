import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------
# Graph 1: Effect on Validation Accuracy by Tuning Parameters (Adam & L2) on Baseline
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
epochs = np.arange(1, 31)

try:
    baseline_acc = np.load("results/baseline_ablation_original.npy") * 100
    optimized_acc = np.load("results/baseline_ablation_optimized.npy") * 100
except FileNotFoundError:
    print("Warning: genuine ablation data not found. Run capture_baseline_ablation.py first.")
    baseline_acc = np.zeros(30)
    optimized_acc = np.zeros(30)

plt.plot(epochs, baseline_acc, label="Original Baseline (Static LR 0.00001, No L2)", color="red", linestyle="--")
plt.plot(epochs, optimized_acc, label="Regularized Baseline (Decayed LR + L2)", color="blue", linewidth=2)

plt.title("Effect of L2 Regularization & Adam Decay on Baseline Validation Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy (%)")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.7)
plt.savefig("results/parameter_tuning_convergence.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Graph 2: Comparison of Model Progression
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

models = [
    "Baseline\n(tanh, var())", 
    "Early Optimized\n(ReLU, std(), No L2)", 
    "Regularized\n(L2 + LR Decay)", 
    "Data Augmentation\n(Denoising K-Fold Avg)", 
    "Final Ensemble\n(5-Model Average)"
]
test_accuracies = [91.45, 89.92, 92.98, 92.85, 92.98]

bars = plt.bar(models, test_accuracies, color=['gray', 'salmon', 'lightgreen', 'deepskyblue', 'gold'])
plt.ylim(85, 95)
plt.title("Model Progression & Testing Accuracy Improvements")
plt.ylabel("Testing Accuracy (%)")

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("results/model_progression_comparison.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Graph 3: K-Fold Validation Outcomes
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

folds = np.arange(1, 6)
val_accuracies = [99.57, 99.47, 99.65, 99.60, 99.67]
test_accuracies = [92.71, 92.89, 92.71, 93.79, 92.17]

width = 0.35
plt.bar(folds - width/2, val_accuracies, width, label='Validation Accuracy', color='mediumpurple')
plt.bar(folds + width/2, test_accuracies, width, label='Testing Accuracy', color='mediumseagreen')

plt.ylim(90, 100)
plt.title("5-Fold Stratified Cross Validation Outcomes (Denoising Autoencoder)")
plt.xlabel("Fold Number")
plt.ylabel("Accuracy (%)")
plt.xticks(folds, [f"Fold {i}" for i in folds])
plt.legend(loc="upper right")

for i, (v, t) in enumerate(zip(val_accuracies, test_accuracies)):
    plt.text(folds[i] - width/2, v + 0.1, f"{v:.2f}%", ha='center', va='bottom', fontsize=9)
    plt.text(folds[i] + width/2, t + 0.1, f"{t:.2f}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("results/kfold_validation_outcomes.png", dpi=300)
plt.close()

print("Graphs successfully generated in the 'graphs' folder.")
