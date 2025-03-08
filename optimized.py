import numpy as np
import pandas as pd
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel  # Added for t-test

# Function to load dataset from a file
def load_dataset(filename):
    return pd.read_csv(filename, header=None)

# Preprocessing function to convert categorical data to numeric using LabelEncoder
def preprocess_data(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col])
    return df

# Euclidean distance calculation (standard for KNN)
def euclidean_distance(row1, row2):
    return sqrt(sum((row1[i] - row2[i]) ** 2 for i in range(len(row1))))

# Calculate local density by measuring the average distance of a point to its k nearest neighbors
def local_density(X_train, k=5):
    densities = []
    
    for i in range(len(X_train)):
        distances = []
        for j in range(len(X_train)):
            if i != j:
                distances.append(euclidean_distance(X_train[i], X_train[j]))
        
        # Sort distances and take the average of the k nearest distances
        distances.sort()
        density = np.mean(distances[:k])  # Average distance to k nearest neighbors
        densities.append(density)
    
    return densities

# DB-kNN Algorithm implementation (density-based KNN)
def db_knn_predict(X_train, y_train, test_instance, k=3, density_weights=None):
    distances = []
    
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], test_instance)
        distances.append((distance, y_train[i], density_weights[i]))
    
    # Sort by distance and get the nearest k neighbors
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    
    # Density-based weighted voting: neighbors with lower density have higher weight
    class_votes = {}
    for dist, label, density in k_neighbors:
        weight = (1 / (dist + 1e-5)) * (1 / (density + 1e-5))  # Inverse distance and inverse density as weight
        class_votes[label] = class_votes.get(label, 0) + weight
    
    # Majority vote
    most_common = max(class_votes, key=class_votes.get)
    return most_common

# K-fold cross-validation implementation from scratch
# Modified to return the list of fold accuracies for later statistical testing.
def k_fold_cross_validation(X, y, k=10, knn_k=3):
    fold_size = len(X) // k
    accuracies = []
    
    for fold in range(k):
        # Create training and testing splits
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_test, y_test = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        # Calculate density weights for the training set
        density_weights = local_density(X_train, k=knn_k)
        
        correct_predictions = 0
        for i in range(len(X_test)):
            prediction = db_knn_predict(X_train, y_train, X_test[i], k=knn_k, density_weights=density_weights)
            if prediction == y_test[i]:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(X_test)
        accuracies.append(accuracy)
    
    return accuracies

# Function to compare DB-kNN with Scikit-Learn's KNN and perform t-test for significance
def compare_knn_implementations(X, y, knn_k=3):
    # DB-kNN implementation (get fold accuracies)
    db_knn_accuracies = k_fold_cross_validation(X.values, y.values, k=10, knn_k=knn_k)
    accuracy_db_knn = np.mean(db_knn_accuracies)
    
    # Scikit-Learn's KNN implementation (get fold accuracies using cross_val_score)
    knn_sklearn = KNeighborsClassifier(n_neighbors=knn_k)
    sklearn_accuracies = cross_val_score(knn_sklearn, X.values, y.values, cv=10)
    accuracy_sklearn = np.mean(sklearn_accuracies)
    
    print(f"DB-kNN Accuracy (with density weighting): {accuracy_db_knn * 100:.2f}%")
    print(f"Scikit-Learn KNN Accuracy: {accuracy_sklearn * 100:.2f}%")
    
    # Perform paired t-test on the fold accuracies
    t_stat, p_value = ttest_rel(db_knn_accuracies, sklearn_accuracies)
    print(f"T-test statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("The difference in performance is statistically significant.")
    else:
        print("The difference in performance is not statistically significant.")

# Main function to load datasets, preprocess, and compare KNN
def main():
    # Load datasets
    breast_cancer_path = 'breast-cancer.data'
    car_path = 'car.data'
    hayes_roth_path = 'hayes-roth.data'
    
    breast_cancer_df = load_dataset(breast_cancer_path)
    car_df = load_dataset(car_path)
    hayes_roth_df = load_dataset(hayes_roth_path)

    # Preprocess datasets (Hayes-Roth is already numeric)
    breast_cancer_df = preprocess_data(breast_cancer_df)
    car_df = preprocess_data(car_df)
    
    # Splitting features and labels for each dataset
    X_bc, y_bc = breast_cancer_df.iloc[:, 1:], breast_cancer_df.iloc[:, 0]
    X_car, y_car = car_df.iloc[:, :-1], car_df.iloc[:, -1]
    X_hr, y_hr = hayes_roth_df.iloc[:, :-1], hayes_roth_df.iloc[:, -1]
    
    # Running the comparison on the datasets
    print("Breast Cancer Dataset:")
    compare_knn_implementations(X_bc, y_bc, knn_k=3)
    
    print("\nCar Evaluation Dataset:")
    compare_knn_implementations(X_car, y_car, knn_k=3)
    
    print("\nHayes-Roth Dataset:")
    compare_knn_implementations(X_hr, y_hr, knn_k=3)

# Entry point of the script
if __name__ == "__main__":
    main()
