import copy
import heapq
import random

allFeatures = {1, 2, 3, 4, 5, 6}

class Node():
    def __init__(self, features = set(), score = 0):
        self.features = features
        self.score = score

    def __str__(self):
        if len(self.features) == 0:
            return f"Using no features and random evaluation, accuracy is {100*self.score}"
        elif len(self.features) == 1:
            return f"Feature {self.features} has accuracy {100*self.score}%"
        else:
            return f"Features {self.features} have accuracy {100*self.score}%"

    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.score < other.score
    
    def addFeature(self, feature):
        self.features.add(feature)
    
    def removeFeature(self, feature):
        self.features.discard(feature)
    
    def cloneAfterAdding(self, feature):
        newFeatures = self.features.copy()
        newFeatures.add(feature)
        return Node(newFeatures, self.score)

    def cloneAfterRemoving(self, feature):
        newFeatures = self.features.copy()
        newFeatures.discard(feature)
        return Node(newFeatures, self.score)

def evaluationFunction(n):
    # STUB
    return round(random.uniform(0, 1), 4)

def expandForward(n):
    expanded = []
    for f in allFeatures:
        if f not in n.features:
           expanded.append(n.cloneAfterAdding(f))
    return expanded