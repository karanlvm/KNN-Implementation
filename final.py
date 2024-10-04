import numpy as np
import pandas as pd
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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
        distances.sort()
        density = np.mean(distances[:k])  # Average distance to k nearest neighbors
        densities.append(density)
    return densities

# KNN Algorithm implementation from scratch
def knn_predict(X_train, y_train, test_instance, k=3):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], test_instance)
        distances.append((distance, y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for _, label in distances[:k]]
    most_common = Counter(k_neighbors).most_common(1)
    return most_common[0][0]

# DB-kNN Algorithm implementation (density-based KNN)
def db_knn_predict(X_train, y_train, test_instance, k=3, density_weights=None):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], test_instance)
        distances.append((distance, y_train[i], density_weights[i]))
    
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    
    class_votes = {}
    for dist, label, density in k_neighbors:
        weight = (1 / (dist + 1e-5)) * (1 / (density + 1e-5))  # Inverse distance and inverse density as weight
        class_votes[label] = class_votes.get(label, 0) + weight
    
    most_common = max(class_votes, key=class_votes.get)
    return most_common

# K-fold cross-validation implementation with parallelization
def k_fold_cross_validation(X, y, k=10, knn_k=3, db_knn=False):
    fold_size = len(X) // k

    def evaluate_fold(fold):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_test, y_test = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        if db_knn:
            density_weights = local_density(X_train, k=knn_k)
            correct_predictions = 0
            for i in range(len(X_test)):
                prediction = db_knn_predict(X_train, y_train, X_test[i], k=knn_k, density_weights=density_weights)
                if prediction == y_test[i]:
                    correct_predictions += 1
        else:
            correct_predictions = 0
            for i in range(len(X_test)):
                prediction = knn_predict(X_train, y_train, X_test[i], k=knn_k)
                if prediction == y_test[i]:
                    correct_predictions += 1
        
        return correct_predictions / len(X_test)

    accuracies = Parallel(n_jobs=-1)(delayed(evaluate_fold)(fold) for fold in range(k))
    return np.mean(accuracies)

# Function to compare custom KNN, DB-kNN, and Scikit-Learn's KNN
def compare_knn_implementations(X, y, knn_k=3):
    accuracy_my_knn = k_fold_cross_validation(X.values, y.values, k=10, knn_k=knn_k)
    accuracy_db_knn = k_fold_cross_validation(X.values, y.values, k=10, knn_k=knn_k, db_knn=True)
    
    knn_sklearn = KNeighborsClassifier(n_neighbors=knn_k)
    accuracy_sklearn = np.mean(cross_val_score(knn_sklearn, X.values, y.values, cv=10))
    
    return accuracy_my_knn, accuracy_db_knn, accuracy_sklearn

# Main function to load datasets, preprocess, and visualize comparison
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
    
    # Collecting accuracies for each dataset
    datasets = ['Breast Cancer', 'Car Evaluation', 'Hayes-Roth']
    accuracies_my_knn = []
    accuracies_db_knn = []
    accuracies_sklearn = []
    
    for X, y in [(X_bc, y_bc), (X_car, y_car), (X_hr, y_hr)]:
        acc_my_knn, acc_db_knn, acc_sklearn = compare_knn_implementations(X, y, knn_k=3)
        accuracies_my_knn.append(acc_my_knn)
        accuracies_db_knn.append(acc_db_knn)
        accuracies_sklearn.append(acc_sklearn)
    
    # Plotting the results
    bar_width = 0.2
    index = np.arange(len(datasets))

    plt.figure(figsize=(10, 6))
    plt.bar(index, accuracies_my_knn, bar_width, label='Custom KNN')
    plt.bar(index + bar_width, accuracies_db_knn, bar_width, label='DB-KNN')
    plt.bar(index + 2 * bar_width, accuracies_sklearn, bar_width, label='Scikit-Learn KNN')
    
    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.title('Comparison of KNN Implementations Across Datasets')
    plt.xticks(index + bar_width, datasets)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()
