import numpy as np
import pandas as pd
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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
# K-fold cross-validation implementation from scratch
def k_fold_cross_validation(X, y, k=10, knn_k=3):
    fold_size = len(X) // k
    accuracies = []
    
    for fold in range(k):
        # Create training and testing splits
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_test, y_test = X[start:end], y.iloc[start:end]  # Use .iloc for Series
        X_train = pd.concat([X[:start], X[end:]])
        y_train = pd.concat([y[:start], y[end:]])
        
        # Calculate density weights for the training set
        density_weights = local_density(X_train.values, k=knn_k)
        
        correct_predictions = 0
        for i in range(len(X_test)):
            prediction = db_knn_predict(X_train.values, y_train.values, X_test.iloc[i], k=knn_k, density_weights=density_weights)
            if prediction == y_test.iloc[i]:  # Use .iloc to access Series
                correct_predictions += 1
        
        accuracy = correct_predictions / len(X_test)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

# Function to compare DB-kNN with Scikit-Learn's KNN
def compare_knn_implementations(X, y, knn_k=3):
    accuracy_db_knn = k_fold_cross_validation(X, y, k=10, knn_k=knn_k)  # Pass X and y directly
    
    # Scikit-Learn's KNN with Grid Search for hyperparameter tuning
    knn_sklearn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
    grid_search = GridSearchCV(knn_sklearn, param_grid, cv=10)
    grid_search.fit(X, y)
    accuracy_sklearn = grid_search.best_score_

    print(f"DB-kNN Accuracy (with density weighting): {accuracy_db_knn * 100:.2f}%")
    print(f"Scikit-Learn KNN Accuracy (best): {accuracy_sklearn * 100:.2f}%")

# Main function to load datasets, preprocess, and compare KNN
# Main function to load datasets, preprocess, and compare KNN
def main():
    breast_cancer_path = 'breast-cancer.data'
    car_path = 'car.data'
    hayes_roth_path = 'hayes-roth.data'
    
    # Load datasets
    breast_cancer_df = load_dataset(breast_cancer_path)
    car_df = load_dataset(car_path)
    hayes_roth_df = load_dataset(hayes_roth_path)

    # Check loaded data
    print("Breast Cancer Dataset:")
    print(breast_cancer_df.head())
    print("\nCar Dataset:")
    print(car_df.head())
    print("\nHayes-Roth Dataset:")
    print(hayes_roth_df.head())

    # Preprocess datasets
    breast_cancer_df = preprocess_data(breast_cancer_df)
    car_df = preprocess_data(car_df)

    # Ensure the DataFrames are not empty
    if breast_cancer_df.empty or car_df.empty or hayes_roth_df.empty:
        print("One or more datasets are empty.")
        return

    # Splitting features and labels for each dataset
    X_bc, y_bc = breast_cancer_df.iloc[:, 1:], breast_cancer_df.iloc[:, 0]
    X_car, y_car = car_df.iloc[:, :-1], car_df.iloc[:, -1]
    X_hr, y_hr = hayes_roth_df.iloc[:, :-1], hayes_roth_df.iloc[:, -1]

    # Standardization
    scaler = StandardScaler()
    X_bc = scaler.fit_transform(X_bc)
    X_car = scaler.fit_transform(X_car)
    X_hr = scaler.fit_transform(X_hr)

    # Optional: Dimensionality Reduction
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_bc = pca.fit_transform(X_bc)
    X_car = pca.fit_transform(X_car)
    X_hr = pca.fit_transform(X_hr)

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
