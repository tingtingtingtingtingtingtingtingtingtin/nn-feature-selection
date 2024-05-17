import copy
import heapq
import random

allFeatures = {1, 2, 3, 4, 5, 6}

class Node():
    def __init__(self, features = set(), score = 0):
        self.features = features
        self.score = score
