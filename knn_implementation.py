"""
Name: Varun Date
UTA ID: 1002198497
Assignment ID: Project 1 - Question 2
"""
import math
import numpy as np


# Defining global variables
POINTS = None
IP_FILE_PATH = "./P1input2024.txt"
OP_FILE_PATH = "./P1Output2024.txt"
LONG_OP_FILE = "./P1input2024LongRecords.txt"
TREES = [(4, 0), (0, 4), (3, 9), (7, 9)]


class KNN:
    """
    Custom implementation of K Nearest Neighbors classifier
    """
    def __init__(self, n_neighbors: int):
        self.k = n_neighbors
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Method to train the KNN classifier on training data

        Args:
            X_train (_type_): training feature vector
            y_train (_type_): training label vector
        """
        self.classes = set(y_train)
        self._mappings = {tuple(sample): label for sample, label in zip(X_train, y_train)}
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Method to predict labels for give data points

        Args:
            X_test (np.ndarray): test feature vector

        Returns:
            np.ndarray: vector of predicted labels
        """
        preds = []

        for vec in X_test:
            # list to keep track of nearest neighbors
            nearest_neighbors  = []

            # get the distance of current point from all points in training data
            for sample in self._mappings:
                d = get_manhattan_distance(vec, sample)
                nearest_neighbors.append((d, sample))

            # sort the nearest neighbors list and get class label from first k elements
            nearest_neighbors.sort(key=lambda x: x[0])
            preds.append(self.get_class_prediction(nearest_neighbors[:self.k]))
        
        return np.array(preds)
    
    def get_class_prediction(self, neighbors: list) -> int:
        """
        Method to vote on class label from the list on nearest neighbors

        Args:
            neighbors (list): list of k nearest neighbors

        Returns:
            int: 0 or 1 class label
        """
        count_1 = 0

        # add labels of all the neighbors
        # if sum is > k / 2, class is 1 else class is 0
        for nei in neighbors:
            sample = nei[1]
            count_1 += self._mappings[sample]
        
        return 1 if count_1 > len(neighbors) / 2 else 0
    
    def get_metrics(self, y_truthy: np.ndarray, y_preds: np.ndarray) -> dict:
        """
        Method to find accuracy, precision, recall, f1 score and confusion matrix for the classifier

        Args:
            y_truthy (np.ndarray): actual labels
            y_preds (np.ndarray): predicted labels

        Returns:
            dict: a dictionary with all the metrics
        """
        tp, fp, tn, fn = 0, 0, 0, 0

        if len(y_truthy) != len(y_preds):
            # to make predictions y_truthy and y_preds should be same length
            return 0.0
        
        # Calculate TP, FP, TN, FN values
        for u, v in zip(y_truthy, y_preds):
            if u == 1:
                tp += 1 if v == 1 else 0
                fn += 1 if v == 0 else 0
            else:
                tn += 1 if v == 0 else 0
                fp += 1 if v == 1 else 0
        
        # Calculate metric values
        results = {
            "accuracy": (tp + tn) / (len(y_preds)),
            "precision": tp / (tp + fn),
            "recall": tp / (tp + fp),
            "f1": (2 * tp) / (2 * tp + fp + fn),        # f1 score = (2 * P * R) / (P + R) = (2 * TP) / (2 * TP + FP + FN)
            "confusion_matrix": [[tn, fp], [fn, tp]]
        }

        return results


def get_manhattan_distance(p1: "tuple", p2: "tuple") -> int:
    """
    Method to find the manhattan distance between two points p1 and p2

    Args:
        p1 (tuple): first data point
        p2 (tuple): second data point

    Returns:
        int: manhattan distance
    """
    if len(p1) != len(p2):
        # Dimensions of the two data points should be same for calculate distance
        return -1
    
    return sum(abs(p1[i] - p2[i]) for i in range(len(p1)))


def get_nearest_tree(x: int, y: int) -> "tuple":
    """
    MEthod to get the nearest tree to the given data point

    Args:
        x (int): x coordinate of the data point
        y (int): y coordinate of the data point

    Returns:
        tuple: Manhattan distance of the tree, tree number
    """
    min_dist, nearest_tree = 1e6, "T1"

    for i, (u, v) in enumerate(TREES):
        d = get_manhattan_distance((u, v), (x, y))

        if d < min_dist:
            min_dist = d
            nearest_tree = f"T{i + 1}"
        
    return min_dist, nearest_tree


def create_columns(data: np.ndarray) -> np.ndarray:
    """
    Method to create the manhattan distance and tree name columns

    Args:
        data (np.ndarray): feature vectors

    Returns:
        np.ndarray: feature vectors appended with manhattan distance and tree name columns
    """
    n = len(data)
    mhd = [0] * n
    tree = [""] * n

    for i, (x, y) in enumerate(data[:, :2]):
        d, t = get_nearest_tree(x, y)

        mhd[i] = d
        tree[i] = t

    mhd = np.array(mhd)
    tree = np.array(tree)
    data = np.concatenate((data, mhd.reshape((-1, 1))), axis=1)
    data = np.concatenate((data, tree.reshape((-1, 1))), axis=1)
    return data


def one_hot_encoding(column: np.ndarray) -> np.ndarray:
    """
    Method to create one hot encoded columns for the categorical feature column

    Args:
        column (np.ndarray): one categorical feature column of the data

    Returns:
        np.ndarray: one hot encoded columns for the given categorical feature
    """
    feature_map = {val: i for i, val in enumerate(np.unique(column))}
    one_hot_encoded_columns = np.zeros((len(column), len(feature_map)), dtype=int)
    
    for i, val in enumerate(column):
        idx = feature_map[val]
        one_hot_encoded_columns[i, idx] = 1
    
    return one_hot_encoded_columns


def train_test_split(X: np.ndarray, y: np.ndarray, train_size: float, shuffle=True, random_state=0) -> "tuple":
    """
    Method to split the data into training and testing samples

    Args:
        X (np.ndarray): feature columns of data
        y (np.ndarray): label columns of data
        train_size (float): percentage of data to be allocated for training
        shuffle (bool, optional): should the inputs be shuffled randomly. Defaults to True.
        random_state (int, optional): for setting the random seed. Defaults to 0.

    Returns:
        tuple: tuple of X_train, X_test, y_train and y_test numpy arrays
    """
    # Combine X and y for shuffling
    training_data = np.concatenate((X, y.reshape((-1, 1))), axis=1)
    
    if random_state:
        # Setting numpy random seed
        np.random.seed(random_state)
    
    if shuffle:
        # Shuffling the data to remove order
        np.random.shuffle(training_data)

    # Calculate and number of training records
    n = len(training_data)
    n_train = math.ceil(n * train_size)
    
    # Separate the data points into X_train, X_test, y_train, y_test
    X_train, X_test = training_data[:n_train, :-1], training_data[n_train:, :-1]
    y_train, y_test = training_data[:n_train, -1], training_data[n_train:, -1]
    return X_train, X_test, y_train, y_test


def main():
    # Open the output file and append Q2
    with open(OP_FILE_PATH, "a") as f:
        f.write("\nQ2\n")
        f.write("-" * 50 + "\n")
    
    feature_columns = {1: 2, 2: 3, 3: 7}    # Mapping of number of feature columns per feature set
    data = np.genfromtxt(IP_FILE_PATH, delimiter="\t", dtype=int)   # Read the data from input file
    
    for x, y in TREES:
        # Adding tree coordinates to the data
        data = np.vstack((data, (x, y, 0)))
    
    # Creating long data records
    long_data = create_columns(data[:, :2])
    ohc = one_hot_encoding(long_data[:, 3])
    long_data = np.concatenate((long_data, ohc), axis=1)
    long_data = np.concatenate((long_data, data[:, 2].reshape((-1, 1))), axis=1)

    # Saving long data to secondary output file
    np.savetxt(LONG_OP_FILE, long_data, delimiter="\t", fmt="%s")

    # Removing categorical 'tree name' columns as we have its info one hot encoded
    data = np.concatenate((long_data[:, :3], long_data[:, 4:]), axis=1).astype(int)

    # Running experiments
    for feature_set in feature_columns:
        result = []     # List to capture outputs
        result.append(f"\nFeature set: {feature_set}\n")
        
        # Getting feature and label columns
        cols = feature_columns[feature_set]
        X, y = data[:, :cols], data[:, -1]

        for k in [3, 5, 7]:
            result.append(f"K:{k}\n")

            # Initializing KNN classifier object
            clf = KNN(n_neighbors=k)

            # Splitting data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

            # Fitting the classifier on training data
            clf.fit(X_train, y_train)

            # Making predictions for test data
            y_preds = clf.predict(X_test)

            # Calculating metrics
            metrics = clf.get_metrics(y_test, y_preds)
            
            for metric, value in metrics.items():
                if metric.strip() == "confusion_matrix":
                    result.append("\n".join([f"{metric}:"] + [f"{row}" for row in value]) + "\n")
                else:
                    result.append(f"{metric}: {value:.4f}\n")
        
        # Printing experiment results to stdout
        for row in result:
            print(row.strip())

        # Writing experiment results to output file
        with open(OP_FILE_PATH, "a") as f:
            f.writelines(result)


if __name__ == "__main__":
    main()
