"""
Name: Varun Date
UTA ID: 1002198497
Assignment ID: Project 1 - Question 1
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Defining global variables
IP_FILE_PATH = "./P1input2024.txt"
OP_FILE_PATH = "./P1Output2024.txt"


def read_input() -> pd.DataFrame:
    """
    Method to read input data from input file

    Returns:
        pd.DataFrame: dataframe of input data
    """
    grid = {"x": [], "y": [], "label": []}
    trees = [(4, 0), (0, 4), (3, 9), (7, 9)]

    # Adding tree coordinates to input data
    for x, y in trees:
        grid["x"].append(x)
        grid["y"].append(y)
        grid["label"].append(0)

    with open(IP_FILE_PATH, "r") as f:
        for line in f:
            x, y, c = map(int, line.split("\t"))            
            grid["x"].append(x)
            grid["y"].append(y)
            grid["label"].append(c)
    
    return pd.DataFrame(grid)


def print_results(k: int, y_true, y_preds):
    """
    Method to print experiment results

    Args:
        k (int): number of neighbors
        y_true (pd.Series): true labels
        y_preds (_type_): predicted labels
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    conf_mat = confusion_matrix(y_true, y_preds, labels=[0, 1])

    output = [
        f"K: {k}\n",
        f"Accuracy: {accuracy:.4f}\n",
        f"Precision: {precision:.4f}\n",
        f"Recall: {recall:.4f}\n",
        f"F1 score: {f1:.4f}\n",
        "Confusion Matrix:\n",
        f"{conf_mat[0]}\n",
        f"{conf_mat[1]}\n"
    ]

    for row in output:
        print(row, end="")
    
    print()

    with open(OP_FILE_PATH, "a") as f:
        f.writelines(output)


def main():
    # Get input data
    df = read_input()

    # Define number of neighbors
    n_neighbors = [3, 5, 7]

    # Split input data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(df[["x", "y"]], df["label"], train_size=0.8, shuffle=True, random_state=42)
    
    # Reset output file
    with open(OP_FILE_PATH, "w") as f:
        f.close()
    
    print("Q1")
    print("-" * 50)

    with open(OP_FILE_PATH, "a") as f:
        f.write("Q1\n")
        f.write("-" * 50 + "\n")

    # Start the experiment
    for k in n_neighbors:
        # Define classifier and fit it on training data
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        # Make predictions on test data
        y_preds = clf.predict(X_test)

        # Print experiment results
        print_results(k, y_test, y_preds)


if __name__ == "__main__":
    main()
