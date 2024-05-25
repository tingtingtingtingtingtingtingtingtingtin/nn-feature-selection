import math
import random
import sys
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
        # Calculate distances for all training data rows against test data
        distances = self.training_data.apply(lambda row: self.__calculate_distance(self.testing_data, row, features), axis=1)
        best_index = distances.idxmin()
        # Return label of closest neighbor and calculated distance
        return self.training_data[0][best_index], distances[best_index]

    def __calculate_distance(self, test, reference, features):
        return math.sqrt(sum((test[f]-reference[f])**2 for f in features))

class Validator:
    def __init__(self, features, dataset, classifier):
        self.feature_set = features
        self.dataset = dataset
        self.classifier = classifier
        pass

    def validate(self):
        num_instances = self.dataset.shape[0]
        num_successes = 0
        for test_instance_id in range(num_instances):
            self.classifier.train(test_instance_id, self.dataset)
            predicted = self.classifier.test(self.feature_set)
            if predicted[0] == self.classifier.testing_data[0]:
                num_successes += 1
        return num_successes/num_instances

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
        if node.score < best.score: print("\nWarning! Accuracy Decreased!")
        else: best = node
        print(f"\nBest: {node}\n")
        neighbors = expandForward(node)
        for n in neighbors:
            n.score = evaluationFunction(n)
            print(f"    {n}")
        node = max(neighbors, key=lambda n: n.score)
    return best

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
    return best

# print("=====FORWARD SEARCH=====")
# print(f"\nFinished Search! Best Feature Set: {forward_search()}")
# print()
# print("=====BACKWARD SEARCH=====")
# print(f"\nFinished Search! Best Feature Set: {backward_search()}")

# print(small_data.head())
# print(small_data.shape)
# print()
# print(large_data.head())
# print(large_data.shape)

# TESTING
feature_set_small = {3, 5, 7}
feature_set_large = {1, 15, 27}
c = Classifier()

print("=====SMALL DATA SET=====")
v1 = Validator(feature_set_small, small_data, c)

s = time.time()
print(f"Feature subset {v1.feature_set} has a score of {v1.validate()}")
e = time.time()
print(f"Completed in {e-s} seconds")


print("=====LARGE DATA SET=====")
v2 = Validator(feature_set_large, large_data, c)

s = time.time()
print(f"Feature subset {v2.feature_set} has a score of {v2.validate()}")
e = time.time()
print(f"Completed in {e-s} seconds")
