import pandas as pd
import numpy as np

def load_and_preprocess_data(train_path="../UJIndoorLoc/trainingData.csv", test_path="../UJIndoorLoc/validationData.csv"):
    # Load training data
    dataset = pd.read_csv(train_path, header=0)
    
    features = np.asarray(dataset.iloc[:,0:520])
    features[features == 100] = -110
    
    labels = np.asarray(dataset["BUILDINGID"].map(str) + dataset["FLOOR"].map(str))
    labels = np.asarray(pd.get_dummies(labels))
    
    # Load testing data
    test_dataset = pd.read_csv(test_path, header=0)
    
    test_features = np.asarray(test_dataset.iloc[:,0:520])
    test_features[test_features == 100] = -110
    
    test_labels = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
    test_labels = np.asarray(pd.get_dummies(test_labels))
    
    # Returning raw, unscaled features so we can scale dynamically in each KFold
    return features, labels, test_features, test_labels
