""""
Machine Learning Assignment -1 ( CSE 6363 )

Implementation of KNN algorithm from scratch with a K-Fold cross validation

Done by Karan Vasudevamurthy (UTA ID: 1002164438)

"""

import numpy as np
import pandas as pd
from math import sqrt

# Function to load data from the dataset

def load_dataset(filename):
    return pd.read_csv(filename, header=None)


# Preprocessing the data to encoding categorical data to numerical data

# Preprocessing the data to encoding categorical data to numerical data
def preprocessing(df):
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = df[col].unique()
            val_map = {val: idx for idx, val in enumerate(unique_vals)}
            df[col] = df[col].map(val_map)
    return df  # Return after processing all columns

    

# Function for Euclidean Distance calculation

def euclidean(row1, row2):
    return sqrt(sum((row1[i] - row2[i]) ** 2 for i in range(len(row1))))


# The KNN function

def knn(X_train, y_train, test_instance, k=3):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean(X_train[i], test_instance)
        distances.append((distance, y_train[i]))
    
    # Sort by distance and get the nearest k neighbors
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]

    # Apply density-based weighting (inverse of distance)
    class_weights = {}
    for dist, label in k_neighbors:
        weight = 1 / (dist + 1e-5)  # Added a small value to avoid division by zero
        if label in class_weights:
            class_weights[label] += weight
        else:
            class_weights[label] = weight

    # Return the class with the highest weighted vote
    return max(class_weights, key=class_weights.get)


# Evaluation metrics

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def other_metrics(y_true, y_pred):
    true_positive = sum((y_true == 1) & (y_pred == 1))
    false_positive = sum((y_true == 0) & (y_pred == 1))
    false_negative = sum((y_true == 1) & (y_pred == 0))
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


# Implementing K fold Cross validation

def k_fold_cross_validation(X, y, k=10, knn_k=3): # Change K value as reuqired. The assignment asked for K value of 10
    fold_size = len(X) // k
    accuracies = []
    
    for fold in range(k):
        # Create training and testing splits
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_test, y_test = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        correct_predictions = 0
        y_pred = []
        for i in range(len(X_test)):
            prediction = knn(X_train, y_train, X_test[i], k=knn_k)
            y_pred.append(prediction)
            if prediction == y_test[i]:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(X_test)
        accuracies.append(accuracy)
    
    return np.mean(accuracies), y_test, y_pred



# Comparisons with scikit
# Function to compare custom KNN with Scikit-learn's KNN using hypothesis testing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats

def compare_knn_implementations(X, y, knn_k=3):
    # My KNN implementation
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    my_knn_accuracies = []
    sklearn_knn_accuracies = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Custom KNN
        y_pred_my_knn = [knn(X_train.values, y_train.values, X_test.iloc[i].values, k=knn_k) for i in range(len(X_test))]
        my_knn_accuracy = accuracy(y_test.values, y_pred_my_knn)
        my_knn_accuracies.append(my_knn_accuracy)

        # Scikit-learn KNN
        knn_sklearn = KNeighborsClassifier(n_neighbors=knn_k)
        knn_sklearn.fit(X_train, y_train)
        sklearn_accuracy = knn_sklearn.score(X_test, y_test)
        sklearn_knn_accuracies.append(sklearn_accuracy)

    # Mean accuracies
    mean_my_knn_acc = np.mean(my_knn_accuracies)
    mean_sklearn_acc = np.mean(sklearn_knn_accuracies)

    # Perform T-test on the fold accuracies
    t_stat, p_value = stats.ttest_rel(my_knn_accuracies, sklearn_knn_accuracies)

    print(f"My KNN Accuracy: {mean_my_knn_acc * 100:.2f}%")
    print(f"Scikit-Learn KNN Accuracy: {mean_sklearn_acc * 100:.2f}%")
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

    if p_value < 0.05:
        print("The difference in accuracies is statistically significant.")
    else:
        print("No significant difference in accuracies.")



def main():

    # Define paths for all datasets
    breast_cancer = 'breast-cancer.data'
    car = 'car.data'
    hayes_roth = 'hayes-roth.data'

    # Loading the datasets
    print("Loading datasets...")
    breast_cancer_df = load_dataset(breast_cancer)
    car_df = load_dataset(car)
    hayes_roth_df = load_dataset(hayes_roth)

    # Preprocessing datasets (Hayes-Roth dataset is already numeric)
    print("Preprocessing datasets...")
    breast_cancer_df = preprocessing(breast_cancer_df)
    car_df = preprocessing(car_df)

    # Split features and labels for each dataset
    print("Splitting features and labels...")
    X_bc, y_bc = breast_cancer_df.iloc[:, 1:], breast_cancer_df.iloc[:, 0]
    X_car, y_car = car_df.iloc[:, :-1], car_df.iloc[:, -1]
    X_hr, y_hr = hayes_roth_df.iloc[:, :-1], hayes_roth_df.iloc[:, -1]

    # Running the comparison on the Breast Cancer dataset
    print("\nBreast Cancer Dataset:")
    compare_knn_implementations(X_bc, y_bc, knn_k=3)
    
    # Running the comparison on the Car dataset
    print("\nCar Evaluation Dataset:")
    compare_knn_implementations(X_car, y_car, knn_k=3)
    
    # Running the comparison on the Hayes-Roth dataset
    print("\nHayes-Roth Dataset:")
    compare_knn_implementations(X_hr, y_hr, knn_k=3)


# Entry point
if __name__ == "__main__":
    main()
