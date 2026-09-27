import pandas as pd
import numpy as np

def load_and_preprocess_data(train_path="../UJIndoorLoc/trainingData.csv", test_path="../UJIndoorLoc/validationData.csv"):
    # Load training data
    dataset = pd.read_csv(train_path, header=0)
    
    features = np.asarray(dataset.iloc[:,0:520])
    features[features == 100] = -110
    
    # Calculate mean and std on training data
    train_mean = features.mean()
    train_std = features.std()
    
    # Fix: use std instead of var for standard scaling
    features = (features - train_mean) / train_std
    
    labels = np.asarray(dataset["BUILDINGID"].map(str) + dataset["FLOOR"].map(str))
    labels = np.asarray(pd.get_dummies(labels))
    
    # Train/Val split
    np.random.seed(42) # Adding seed for reproducibility
    train_val_split = np.random.rand(len(features)) < 0.70
    train_x = features[train_val_split]
    train_y = labels[train_val_split]
    val_x = features[~train_val_split]
    val_y = labels[~train_val_split]
    
    # Load testing data
    test_dataset = pd.read_csv(test_path, header=0)
    
    test_features = np.asarray(test_dataset.iloc[:,0:520])
    test_features[test_features == 100] = -110
    
    # Apply training mean and std to test set
    test_features = (test_features - train_mean) / train_std
    
    test_labels = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
    test_labels = np.asarray(pd.get_dummies(test_labels))
    
    return train_x, train_y, val_x, val_y, test_features, test_labels
