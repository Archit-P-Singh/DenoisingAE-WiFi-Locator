import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from data_loader import load_and_preprocess_data
from autoencoder import AutoEncoderNN

def main():
    print("Loading data...")
    train_x, train_y, val_x, val_y, test_features, test_labels = load_and_preprocess_data()
    
    n_input = 520
    n_classes = train_y.shape[1]
    
    # Updated parameters for stability
    learning_rate = 0.0005
    training_epochs = 60
    batch_size = 32
    total_batches = train_x.shape[0] // batch_size
    
    X = tf.placeholder(tf.float32, shape=[None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    # Initialize Model
    model = AutoEncoderNN(n_input, n_classes)
    
    encoded = model.encode(X)
    decoded = model.decode(encoded)
    y_ = model.dnn(encoded, keep_prob)
    
    # Loss functions and optimizers
    us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
    # Epsilon added to avoid log(0) issues which often crash TF during cross entropy
    s_cost_function = -tf.reduce_sum(Y * tf.log(y_ + 1e-10))
    
    us_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(us_cost_function)
    s_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(s_cost_function)
    
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Training
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        
        # ------------ 1. Training Autoencoders - Unsupervised Learning ----------- #
        print("Starting Unsupervised Pre-training...")
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                _, c = session.run([us_optimizer, us_cost_function], feed_dict={X: batch_x})
                epoch_costs = np.append(epoch_costs, c)
            print(f"Epoch: {epoch}  Loss: {np.mean(epoch_costs):.4f}")
        print("Unsupervised pre-training finished...\n")
        
        # ---------------- 2. Training NN - Supervised Learning ------------------ #
        print("Starting Supervised Training...")
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                batch_y = train_y[offset:(offset + batch_size), :]
                _, c = session.run([s_optimizer, s_cost_function], feed_dict={X: batch_x, Y : batch_y, keep_prob: 0.8})
                epoch_costs = np.append(epoch_costs, c)
            
            train_acc = session.run(accuracy, feed_dict={X: train_x, Y: train_y, keep_prob: 1.0})
            val_acc = session.run(accuracy, feed_dict={X: val_x, Y: val_y, keep_prob: 1.0})
            print(f"Epoch: {epoch}  Loss: {np.mean(epoch_costs):.4f}  Training Accuracy: {train_acc:.4f}  Validation Accuracy: {val_acc:.4f}")
                
        print("Supervised training finished...\n")
        
        # Testing
        test_acc = session.run(accuracy, feed_dict={X: test_features, Y: test_labels, keep_prob: 1.0})
        print(f"Testing Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
