import copy
import queue
import random

allFeatures = {1, 2, 3, 4, 5, 6}

class Node():
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
    startNode = Node()
    startNode.score = evaluationFunction(startNode)
    q = queue.Queue()
    q.put(startNode)
    best = startNode
    print("Beginning Search...")
    while q.not_empty:
        node = q.get()
        if node.score < best.score: print("\nWarning! Accuracy Decreased!")
        else: best = node
        if node.features == allFeatures: 
            print(f"Finished Search! The best {best}")
            return
        print(f"Best: {node}")
        print()
        neighbors = expandForward(node)
        for n in neighbors:
            n.score = evaluationFunction(n)
            print(n)
        q.put(max(neighbors, key=lambda n: n.score))

forward_search()