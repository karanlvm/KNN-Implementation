import numpy as np
import pandas as pd
from math import sqrt

# Function to load dataset from a file
def load_dataset(filename):
    return pd.read_csv(filename, header=None)

# Preprocessing function to manually encode categorical data to numeric
def preprocess_data(df):
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = df[col].unique()
            val_map = {val: idx for idx, val in enumerate(unique_vals)}
            df[col] = df[col].map(val_map)
    return df

# Euclidean distance calculation
def euclidean_distance(row1, row2):
    return sqrt(sum((row1[i] - row2[i]) ** 2 for i in range(len(row1))))

# Majority vote function
def majority_vote(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        if neighbor in class_votes:
            class_votes[neighbor] += 1
        else:
            class_votes[neighbor] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda item: item[1], reverse=True)
    return sorted_votes[0][0]

# KNN Algorithm implementation from scratch
def knn_predict(X_train, y_train, test_instance, k=3):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], test_instance)
        distances.append((distance, y_train[i]))
    
    # Sort by distance and get the nearest k neighbors
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for _, label in distances[:k]]
    
    # Majority vote
    return majority_vote(k_neighbors)

# K-fold cross-validation implementation from scratch
def k_fold_cross_validation(X, y, k=10, knn_k=3):
    fold_size = len(X) // k
    accuracies = []
    
    for fold in range(k):
        # Create training and testing splits
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_test, y_test = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        correct_predictions = 0
        for i in range(len(X_test)):
            prediction = knn_predict(X_train, y_train, X_test[i], k=knn_k)
            if prediction == y_test[i]:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(X_test)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

# Custom accuracy metric calculation
def calculate_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

# Function to compare custom KNN with Scikit-Learn's KNN
def compare_knn_implementations(X, y, knn_k=3):
    # My KNN implementation
    accuracy_my_knn = k_fold_cross_validation(X.values, y.values, k=10, knn_k=knn_k)
    
    # Scikit-Learn's KNN implementation (only used for comparison)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    knn_sklearn = KNeighborsClassifier(n_neighbors=knn_k)
    accuracy_sklearn = np.mean(cross_val_score(knn_sklearn, X.values, y.values, cv=10))
    
    print(f"My KNN Accuracy: {accuracy_my_knn * 100:.2f}%")
    print(f"Scikit-Learn KNN Accuracy: {accuracy_sklearn * 100:.2f}%")

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
