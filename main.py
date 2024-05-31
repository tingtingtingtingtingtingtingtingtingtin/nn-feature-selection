import time
import numpy as np
import pandas as pd

# Training data
small = 'small-test-dataset.txt'
num_features_small = 10
large = 'large-test-dataset.txt'
num_features_large = 40


# Load data from user input
def acquireData():
    print("Which file would you like to use?")
    print("1. Small Test Dataset")
    print("2. Large Test Dataset")
    print("3. Custom")
    choice = int(input("Enter Selection: "))
    file = ""
    if choice == 1:
        file = "small-test-dataset.txt"
    if choice == 2:
        file = "large-test-dataset.txt"
    if choice == 3:
        file = input("Enter the file you would like to test: ")
    try:
        data = pd.read_csv(file, sep="\\s+", header=None)
        return data
    except FileNotFoundError:
        print("Invalid File Name!")
        exit(1)

class Node:
    def __init__(self, features = set(), score = 0):
        self.features = features
        self.score = score

    def __str__(self):
        if len(self.features) == 0:
            return f"Feature {{}} has accuracy {round(self.score, 2)}%"
        elif len(self.features) == 1:
            return f"Feature {self.features} has accuracy {round(self.score, 2)}%"
        else:
            return f"Features {self.features} have accuracy {round(self.score, 2)}%"
    
    def cloneAfterAdding(self, feature):
        newFeatures = self.features.copy()
        newFeatures.add(feature)
        return Node(newFeatures, self.score)

    def cloneAfterRemoving(self, feature):
        newFeatures = self.features.copy()
        newFeatures.discard(feature)
        return Node(newFeatures, self.score)

class Classifier:
    def __init__(self):
        self.training_data = None
        self.testing_data = None
    
    def train(self, id, data):
        if id in data.index:
            self.training_data = data.drop(id)
            self.testing_data = data.iloc[id]
    
    def test(self, features):
        if not features:
            return self.training_data[0].value_counts().idxmax(), None, None
        # Calculate distances for all training data rows against test data
        distances = np.sqrt(np.sum((self.testing_data[list(features)].values - self.training_data.values[:, list(features)])**2, axis = 1))
        best_index = np.argmin(distances)
        # Return predicted label and index of closest neighbor
        return self.training_data.iloc[best_index, 0], best_index, distances[best_index]

class Validator:
    def __init__(self, features, dataset, classifier):
        self.feature_set = features
        self.dataset = dataset
        self.classifier = classifier

    def validate(self):
        num_instances = self.dataset.shape[0]
        num_successes = 0
        s = time.time()
        for test_instance_id in range(num_instances):
            # Train and test the classifier on the given instance
            self.classifier.train(test_instance_id, self.dataset)
            predicted = self.classifier.test(self.feature_set)
            # Print trace, should comment out when running search for readability
            # print(f"Instance {test_instance_id} is class {predicted[0]}")
            # print(f"Its nearest neighbor is {predicted[1]} which is of class {predicted[0]}. The distance is {predicted[2]}")
            if predicted[0] == self.classifier.testing_data[0]:
                num_successes += 1
        e = time.time()
        # Also can be commented out for readability
        # print(f"\nDone! Validated {num_instances} instances in {e-s} seconds.")
        return num_successes/num_instances

def normalize(dataset):
    dataset[dataset.columns[1:]] = dataset[dataset.columns[1:]].apply(lambda col: (col - col.mean())/col.std())

def evaluationFunction(n, v):
    v.feature_set = n.features
    return 100*v.validate()

def expandForward(n):
    return [n.cloneAfterAdding(f) for f in features if f not in n.features]
    # expanded = []
    # for f in allFeatures:
    #     if f not in n.features:
    #        expanded.append(n.cloneAfterAdding(f))
    # return expanded

def expandBackward(n):
    return [n.cloneAfterRemoving(f) for f in features if f in n.features]
    # expanded = []
    # for f in allFeatures:
    #     if f in n.features:
    #         expanded.append(n.cloneAfterRemoving(f))
    # return expanded

def forward_search():
    node = Node()
    c = Classifier()
    validator = Validator(node.features, data, c)
    node.score = evaluationFunction(node, validator)
    best = node
    print("Beginning Search...")
    while node.features != features:
        if node.score <= best.score: print("\nWarning! Accuracy Decreased!")
        else: best = node
        print(f"\nBest: {node}\n")
        neighbors = expandForward(node)
        for n in neighbors:
            n.score = evaluationFunction(n, validator)
            print(f"    {n}")
        node = max(neighbors, key=lambda n: n.score)
    return best if node.score <= best.score else node

def backward_search():
    node = Node(features)
    c = Classifier()
    validator = Validator(node.features, data, c)
    node.score = evaluationFunction(node, validator)
    best = node
    print("Beginning Search...")
    while len(node.features):
        if node.score < best.score: print("\nWarning! Accuracy Decreased!")
        else: best = node
        print(f"\nBest: {node}\n")
        neighbors = expandBackward(node)
        for n in neighbors:
            n.score = evaluationFunction(n, validator)
            print(f"    {n}")
        node = (max(neighbors, key=lambda x: x.score))
    return best if node.score < best.score else node

# Function for testing classifier + validator
def nn_test():
    print("Which of the following feature sets would you like to use?")
    print(f"1. {{3, 5, 7}}")
    print(f"2. {{1, 15, 27}}")
    set_select = input("Enter Selection: ")
    feature_set = {}
    if set_select == "1":
        feature_set = {3, 5, 7}
    elif set_select == "2":
        feature_set = {1, 15, 27}
    c = Classifier()
    v = Validator(feature_set, data, c)
    print(f"\nValidating NN-Classifier using features {v.feature_set}")
    print(f"Feature subset {v.feature_set} has an accuracy of {v.validate()}")

print("====================Welcome to Komay and friends' Feature Search Selection!====================")
print("Featuring:\n\n\tAdithya Iyer (aiyer026)\n\tAndy Jarean (ajare002)\n\tKomay Sugiyama (ksugi014)\n\tTingxuan Wu (twu148)\n")

# Get data from user specified file and extract attributes
data = acquireData()
num_instances = data.shape[0]
num_features = data.shape[1]
features = set(range(1, num_features))
print(f"This dataset has {num_instances} instances and {num_features-1} features (excluding class label).\n")

print("Normalizing data using Z-Score...\n")
normalize(data)

### CODE FOR TEST HARNESS
test_input = input("\n!!!!!!!!!!!!!!!!!!!!\nFOR TESTING PURPOSES, PLEASE INPUT 1 TO TEST NN-CLASSIFIER & VALIDATOR WITH A SPECIFIC FEATURE SET :]\nPRESS ANY OTHER INPUT TO CONTINUE\n!!!!!!!!!!!!!!!!!!!!\n")
if test_input == "1":
    nn_test()
    exit(0)

print("Search Algorithms:\n\n\t1. Forward Selection\n\t2. Backward Elimination\n\t3. Bertie's Special Algorithm\n")
search_type = int(input("Enter the number for the search you would like to run: "))

if search_type == 1:
    print("=====FORWARD SEARCH=====")
    s = time.time()
    f_set = forward_search()
    e = time.time()
    print(f"\nFinished search in {e-s} seconds! Best Feature Set: {f_set}")
elif search_type == 2:
    print("=====BACKWARD SEARCH=====")
    s = time.time()
    f_set = backward_search()
    e = time.time()
    print(f"\nFinished search in {e-s} seconds!! Best Feature Set: {f_set}")
else:
    print("\nIncorrect input. Terminating...\n")
