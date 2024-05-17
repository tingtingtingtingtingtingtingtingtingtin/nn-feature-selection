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
    

