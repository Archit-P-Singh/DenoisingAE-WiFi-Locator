import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'optimized'))
from data_loader import load_and_preprocess_data
from autoencoder import AutoEncoderNN

def run_experiment(use_decay=False, learning_rate_start=0.001, beta=0.0):
    tf.reset_default_graph()
    
    # We must load the data manually so we don't trip over paths
    import pandas as pd
    train_path = 'UJIndoorLoc/trainingData.csv'
    dataset = pd.read_csv(train_path, header=0)
    features = dataset.iloc[:, 0:520].values
    labels_b = pd.get_dummies(dataset['BUILDINGID'])
    labels_f = pd.get_dummies(dataset['FLOOR'])
    labels = pd.concat([labels_b, labels_f], axis=1).values
    
    split = int(0.8 * features.shape[0])
    train_x = features[:split]
    train_y = labels[:split]
    
    mean_val = np.mean(train_x, axis=0)
    std_val = np.std(train_x, axis=0)
    std_val[std_val == 0] = 1
    train_x = (train_x - mean_val) / std_val
    
    n_input = 520
    n_classes = labels.shape[1]
    
    X = tf.placeholder("float", [None, n_input])
    X_clean = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    
    # Noise injection for Denoising
    noise_factor = 0.2
    
    model = AutoEncoderNN(n_input, n_classes)
    
    # 1. Unsupervised Pre-training Phase
    decoder_op = model.decode(model.encode(X))
    us_cost_function = tf.reduce_mean(tf.pow(X_clean - decoder_op, 2))
    us_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(us_cost_function)
    
    # 2. Supervised Training Phase
    classifier_op = model.dnn(model.encode(X), 1.0)
    s_cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=classifier_op))
    
    if beta > 0.0:
        l2_loss = tf.nn.l2_loss(model.e_weights_h1) + tf.nn.l2_loss(model.e_weights_h2) + \
                  tf.nn.l2_loss(model.e_weights_h3) + tf.nn.l2_loss(model.dnn_weights_h1) + \
                  tf.nn.l2_loss(model.dnn_weights_h2)
        s_cost_function = s_cost_function + beta * l2_loss
        
    global_step = tf.Variable(0, trainable=False)
    if use_decay:
        learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 100, 0.95, staircase=True)
    else:
        learning_rate = learning_rate_start
        
    s_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(s_cost_function, global_step=global_step)
    
    init = tf.global_variables_initializer()
    
    epochs = 30
    batch_size = 64
    total_batches = int(train_x.shape[0] / batch_size)
    
    supervised_losses = []
    
    with tf.Session() as session:
        session.run(init)
        
        # Unsupervised
        for epoch in range(epochs):
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                noisy_batch_x = batch_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)
                session.run([us_optimizer, us_cost_function], feed_dict={X: noisy_batch_x, X_clean: batch_x})
                
        # Supervised
        for epoch in range(epochs):
            epoch_costs = []
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                batch_y = train_y[offset:(offset + batch_size), :]
                _, c = session.run([s_optimizer, s_cost_function], feed_dict={X: batch_x, Y: batch_y})
                epoch_costs.append(c)
            supervised_losses.append(np.mean(epoch_costs))
            
    return supervised_losses

print("Running experiment: Bad Settings (LR=0.001, No L2)")
bad_losses = run_experiment(use_decay=False, learning_rate_start=0.001, beta=0.0)

print("Running experiment: Good Settings (LR=0.0005 + Decay, L2=0.001)")
good_losses = run_experiment(use_decay=True, learning_rate_start=0.0005, beta=0.001)

np.save("bad_losses.npy", bad_losses)
np.save("good_losses.npy", good_losses)
print("Saved losses to numpy files.")
