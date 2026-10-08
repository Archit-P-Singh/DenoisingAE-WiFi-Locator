import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import sys

# Important: Use the baseline data loader and autoencoder
sys.path.append(os.path.join(os.path.dirname(__file__), 'baseline'))
from data_loader import load_and_preprocess_data
from autoencoder import AutoEncoderNN

def run_baseline_ablation(apply_optimizations=False):
    tf.reset_default_graph()
    
    # 1. Load Data
    train_x, train_y, val_x, val_y, test_features, test_labels = load_and_preprocess_data(train_path="UJIndoorLoc/trainingData.csv", test_path="UJIndoorLoc/validationData.csv")
    
    n_input = 520
    n_classes = train_y.shape[1]
    
    training_epochs = 30
    batch_size = 15
    total_batches = train_x.shape[0] // batch_size
    
    X = tf.placeholder(tf.float32, shape=[None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    
    # 2. Model (Baseline uses Tanh, standard Autoencoder)
    model = AutoEncoderNN(n_input, n_classes)
    encoded = model.encode(X)
    decoded = model.decode(encoded)
    y_ = model.dnn(encoded)
    
    # 3. Loss Functions
    us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
    s_cost_function = -tf.reduce_sum(Y * tf.log(y_ + 1e-10))
    
    # Apply L2 Regularization if testing optimizations
    if apply_optimizations:
        beta = 0.001
        l2_loss = tf.nn.l2_loss(model.dnn_weights_h1) + tf.nn.l2_loss(model.dnn_weights_h2) + tf.nn.l2_loss(model.dnn_weights_out)
        s_cost_function = s_cost_function + beta * l2_loss
        
    global_step = tf.Variable(0, trainable=False)
    
    # Apply Adam Learning Rate Decay if testing optimizations
    if apply_optimizations:
        learning_rate = tf.train.exponential_decay(0.0005, global_step, decay_steps=total_batches, decay_rate=0.95, staircase=True)
    else:
        # Baseline paper used 0.00001 for Adam
        learning_rate = 0.00001
        
    us_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(us_cost_function)
    s_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(s_cost_function, global_step=global_step)
    
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    val_accuracies = []
    
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        
        # Unsupervised Pre-training
        for epoch in range(training_epochs):
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                session.run([us_optimizer, us_cost_function], feed_dict={X: batch_x})
                
        # Supervised Training
        for epoch in range(training_epochs):
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                batch_y = train_y[offset:(offset + batch_size), :]
                session.run([s_optimizer, s_cost_function], feed_dict={X: batch_x, Y: batch_y})
                
            # Track validation accuracy
            val_acc = session.run(accuracy, feed_dict={X: val_x, Y: val_y})
            val_accuracies.append(val_acc)
            
    return val_accuracies

print("Running pure Baseline (LR=0.00001, No L2, Tanh)...")
baseline_val_acc = run_baseline_ablation(apply_optimizations=False)

print("Running Baseline with optimizations (Decayed Adam LR, L2, Tanh)...")
optimized_val_acc = run_baseline_ablation(apply_optimizations=True)

np.save("results/baseline_ablation_original.npy", baseline_val_acc)
np.save("results/baseline_ablation_optimized.npy", optimized_val_acc)
print("Saved ablation accuracies.")
