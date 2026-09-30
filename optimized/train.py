import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.model_selection import KFold
from data_loader import load_and_preprocess_data
from autoencoder import AutoEncoderNN

def main():
    print("Loading data...")
    features, labels, test_features_raw, test_labels = load_and_preprocess_data()
    
    n_input = 520
    n_classes = labels.shape[1]
    
    # Updated parameters for stability
    initial_learning_rate = 0.0005
    training_epochs = 40 # slightly lowered for faster 5-fold cross validation
    batch_size = 32
    beta = 0.001
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold = 1
    cv_val_scores = []
    cv_test_scores = []
    
    # Store predictions from all 5 models to create an ensemble
    ensemble_test_preds = np.zeros((test_features_raw.shape[0], n_classes))
    
    for train_index, val_index in kf.split(features):
        print(f"\n================ Starting Fold {fold} ================")
        tf.reset_default_graph()
        
        train_x_raw, val_x_raw = features[train_index], features[val_index]
        train_y, val_y = labels[train_index], labels[val_index]
        
        # Scale dynamically to prevent data leakage from validation/test into training
        train_mean = train_x_raw.mean()
        train_std = train_x_raw.std()
        
        train_x = (train_x_raw - train_mean) / train_std
        val_x = (val_x_raw - train_mean) / train_std
        test_features = (test_features_raw - train_mean) / train_std
        
        total_batches = train_x.shape[0] // batch_size
        
        X = tf.placeholder(tf.float32, shape=[None, n_input])
        X_clean = tf.placeholder(tf.float32, shape=[None, n_input])
        Y = tf.placeholder(tf.float32, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)
        
        # Initialize Model
        model = AutoEncoderNN(n_input, n_classes)
        
        encoded = model.encode(X)
        decoded = model.decode(encoded)
        y_ = model.dnn(encoded, keep_prob)
        
        # Loss functions and optimizers
        # Denoising target: compare decoded output to the ORIGINAL (clean) X, but we'll feed noisy X
        us_cost_function = tf.reduce_mean(tf.pow(X_clean - decoded, 2))
        
        cross_entropy = -tf.reduce_sum(Y * tf.log(y_ + 1e-10))
        l2_loss = tf.nn.l2_loss(model.dnn_weights_h1) + tf.nn.l2_loss(model.dnn_weights_h2) + tf.nn.l2_loss(model.dnn_weights_out)
        s_cost_function = cross_entropy + beta * l2_loss
        
        # Learning Rate Decay
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps=total_batches, decay_rate=0.95, staircase=True)
        
        us_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(us_cost_function)
        s_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(s_cost_function, global_step=global_step)
        
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
                    # Add Gaussian noise for Denoising Autoencoder (Data Augmentation)
                    noise_factor = 0.2
                    noisy_batch_x = batch_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)
                    
                    _, c = session.run([us_optimizer, us_cost_function], feed_dict={X: noisy_batch_x, X_clean: batch_x})
                    epoch_costs = np.append(epoch_costs, c)
                # Print every 10 epochs to save space
                if epoch % 10 == 0:
                    print(f"Epoch: {epoch}  Loss: {np.mean(epoch_costs):.4f}")
            
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
                
                # Print every 10 epochs
                if epoch % 10 == 0:
                    train_acc = session.run(accuracy, feed_dict={X: train_x, Y: train_y, keep_prob: 1.0})
                    val_acc = session.run(accuracy, feed_dict={X: val_x, Y: val_y, keep_prob: 1.0})
                    print(f"Epoch: {epoch}  Loss: {np.mean(epoch_costs):.4f}  Training Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")
                    
            final_val_acc = session.run(accuracy, feed_dict={X: val_x, Y: val_y, keep_prob: 1.0})
            final_test_acc = session.run(accuracy, feed_dict={X: test_features, Y: test_labels, keep_prob: 1.0})
            
            print(f"Fold {fold} Finished. Validation Accuracy: {final_val_acc:.4f}, Testing Accuracy: {final_test_acc:.4f}")
            
            cv_val_scores.append(final_val_acc)
            cv_test_scores.append(final_test_acc)
            
            # Save probabilities for the ensemble
            fold_test_preds = session.run(y_, feed_dict={X: test_features, keep_prob: 1.0})
            ensemble_test_preds += fold_test_preds / 5.0
            
        fold += 1
        
    # Calculate Ensemble Accuracy
    ensemble_correct_prediction = np.equal(np.argmax(ensemble_test_preds, axis=1), np.argmax(test_labels, axis=1))
    ensemble_accuracy = np.mean(ensemble_correct_prediction.astype(float))
        
    print("\n================ FINAL K-FOLD & ENSEMBLE RESULTS ================")
    print(f"Average Validation Accuracy across 5 folds: {np.mean(cv_val_scores):.4f} (Std: {np.std(cv_val_scores):.4f})")
    print(f"Average Testing Accuracy across 5 folds: {np.mean(cv_test_scores):.4f} (Std: {np.std(cv_test_scores):.4f})")
    print(f"Ensemble Testing Accuracy (Averaged Softmax over 5 models): {ensemble_accuracy:.4f}")

if __name__ == "__main__":
    main()
