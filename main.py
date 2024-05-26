import math
import random
import time
import pandas as pd

allFeatures = {1, 2, 3, 4, 5, 6}

# Training data
small = 'small-test-dataset.txt'
num_features_small = 10
large = 'large-test-dataset.txt'
num_features_large = 40


# Load training data
small_data = pd.read_csv(small, sep="\\s+", header=None)
large_data = pd.read_csv(large, sep="\\s+", header=None)

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
        distances = self.training_data.apply(lambda row: self.__calculate_distance(self.testing_data, row, features), axis=1)
        best_index = distances.idxmin()
        # Return predicted label and index of closest neighbor
        return self.training_data[0][best_index], best_index, distances[best_index]

    def __calculate_distance(self, test, reference, features):
        return math.sqrt(sum((test[f]-reference[f])**2 for f in features))

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
            # Print trace
            print(f"Instance {test_instance_id} is class {predicted[0]}")
            print(f"Its nearest neighbor is {predicted[1]} which is of class {predicted[0]}. The distance is {predicted[2]}")
            if predicted[0] == self.classifier.testing_data[0]:
                num_successes += 1
        e = time.time()
        print(f"\nDone! Validated {num_instances} instances in {e-s} seconds.")
        return num_successes/num_instances

def normalize(dataset):
    dataset[dataset.columns[1:]] = dataset[dataset.columns[1:]].apply(lambda col: (col - col.mean())/col.std())

def evaluationFunction(n):
    # STUB
    return 100*random.uniform(0, 1)

def expandForward(n):
    return [n.cloneAfterAdding(f) for f in allFeatures if f not in n.features]
    # expanded = []
    # for f in allFeatures:
    #     if f not in n.features:
    #        expanded.append(n.cloneAfterAdding(f))
    # return expanded

def expandBackward(n):
    return [n.cloneAfterRemoving(f) for f in allFeatures if f in n.features]
    # expanded = []
    # for f in allFeatures:
    #     if f in n.features:
    #         expanded.append(n.cloneAfterRemoving(f))
    # return expanded

def forward_search():
    node = Node()
    node.score = evaluationFunction(node)
    best = node
    print("Beginning Search...")
    while node.features != allFeatures:
        if node.score <= best.score: print("\nWarning! Accuracy Decreased!")
        else: best = node
        print(f"\nBest: {node}\n")
        neighbors = expandForward(node)
        for n in neighbors:
            n.score = evaluationFunction(n)
            print(f"    {n}")
        node = max(neighbors, key=lambda n: n.score)
    return best if node.score <= best.score else node

def backward_search():
    node = Node(allFeatures)
    node.score = evaluationFunction(node)
    best = node
    print("Beginning Search...")
    while len(node.features):
        if node.score < best.score: print("\nWarning! Accuracy Decreased!")
        else: best = node
        print(f"\nBest: {node}\n")
        neighbors = expandBackward(node)
        for n in neighbors:
            n.score = evaluationFunction(n)
            print(f"    {n}")
        node = (max(neighbors, key=lambda x: x.score))
    return best if node.score < best.score else node

def nn_test():
    # TESTING
    feature_set_small = {3, 5, 7}
    feature_set_large = {1, 15, 27}
    c = Classifier()

    set_select = input("1. SMALL DATA SET\n2. LARGE DATA SET\n")
    if set_select == "1":
        normalize(small_data)
        v1 = Validator(feature_set_small, small_data, c)
        print(f"=====SMALL DATA SET=====\nUsing features {v1.feature_set}")
        print(f"Feature subset {v1.feature_set} has a score of {v1.validate()}")
    elif set_select == "2":
        normalize(large_data)
        v2 = Validator(feature_set_large, large_data, c)
        print(f"=====LARGE DATA SET=====\nUsing features {v2.feature_set}")
        print(f"Feature subset {v2.feature_set} has a score of {v2.validate()}")
    else:
        print(f"INCORRECT INPUT. ")
    quit()

print("====================\n Welcome to Komay and friends' Feature Search Selection!\n")
print("Featuring:\nAdithya Iyer (aiyer026)\nAndy Jarean (ajare002)\nKomay Sugiyama (ksugi014)\nTingxuan Wu (twu148)\n===================")

test_input = input("\n!!!!!!!!!!!!!!!!!!!!\nFOR TESTING PURPOSES, PLEASE INPUT 1 TO TEST NN-CLASSIFIER & VALIDATOR :]\nPRESS ANY OTHER INPUT TO CONTINUE\n!!!!!!!!!!!!!!!!!!!!\n")
if test_input == "1":
    nn_test()
num_feature_input = int(input("\nPlease enter total number of features: "))
allFeatures = set(range(1, num_feature_input + 1))

print("Features:\n\t1. Forward Selection\n\t2. Backward Elimination\n\t3. Bertie's Special Algorithm")
type_feature_input = int(input("\nType the number of the features you want to run: "))

if type_feature_input == 1:
    print("=====FORWARD SEARCH=====")
    s = time.time()
    f_set = forward_search()
    e = time.time()
    print(f"\nFinished search in {e-s} seconds! Best Feature Set: {f_set}")
elif type_feature_input == 2:
    print("=====BACKWARD SEARCH=====")
    s = time.time()
    f_set = forward_search()
    e = time.time()
    print(f"\nFinished search in {e-s} seconds!! Best Feature Set: {f_set}")
else:
    print("\nIncorrect input. Terminating...\n")
