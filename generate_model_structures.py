import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, GaussianNoise
from tensorflow.keras.utils import plot_model
import os

os.makedirs("graphs", exist_ok=True)

# ---------------------------------------------------------
# 1. Denoising Autoencoder Structure
# ---------------------------------------------------------
ae_input = Input(shape=(520,), name="WiFi_RSSI_Input")
noisy_input = GaussianNoise(0.2, name="Gaussian_Noise_Injection")(ae_input)

# Encoder
encoded = Dense(256, activation='relu', name="Encoder_Hidden_1")(noisy_input)
encoded = Dense(128, activation='relu', name="Encoder_Hidden_2")(encoded)
encoded = Dense(64, activation='relu', name="Bottleneck_Latent_Space")(encoded)

# Decoder
decoded = Dense(128, activation='relu', name="Decoder_Hidden_1")(encoded)
decoded = Dense(256, activation='relu', name="Decoder_Hidden_2")(decoded)
decoded = Dense(520, activation='linear', name="Reconstructed_Clean_Output")(decoded)

autoencoder = Model(inputs=ae_input, outputs=decoded, name="Denoising_Autoencoder")
plot_model(autoencoder, to_file="graphs/Denoising_AE_Structure.png", show_shapes=True, show_layer_names=False, rankdir='LR', dpi=300)

# ---------------------------------------------------------
# 2. Final Ensembled Classifier NN Structure
# ---------------------------------------------------------
nn_input = Input(shape=(64,), name="Latent_Space_Features")

# Classifier
dense1 = Dense(128, activation='relu', name="Classifier_Hidden_1")(nn_input)
dense2 = Dense(128, activation='relu', name="Classifier_Hidden_2")(dense1)
output = Dense(118, activation='softmax', name="Building_Floor_Prediction")(dense2)

classifier = Model(inputs=nn_input, outputs=output, name="Optimized_Neural_Network")
plot_model(classifier, to_file="graphs/Optimized_NN_Structure.png", show_shapes=True, show_layer_names=False, rankdir='LR', dpi=300)

print("Model structure graphs successfully generated in the 'graphs' folder.")
