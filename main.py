import copy
import queue
import random
import tkinter
from tkinter import scrolledtext
import customtkinter
from customtkinter import *
from PIL import Image
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter

#-----------------------------
#UI
#-----------------------------
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

class TextFrame(customtkinter.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.text_widget = customtkinter.CTkTextbox(master=self, width=400, height=600, corner_radius=0, border_color="#99AAB5", border_width=1)
        self.text_widget.grid(row=1, column=0, sticky="nsew")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("NN Feature Selection")
        self.geometry("900x700")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.text_frame = TextFrame(self, title="NN Feature Selection")
        self.text_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew", rowspan=3)

        sys.stdout = TextRedirector(self.text_frame.text_widget)

        self.frame = customtkinter.CTkFrame(self, fg_color="#23272A", border_color="#99AAB5", border_width=1)
        self.frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        self.frame.grid_rowconfigure(15, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        self.image = customtkinter.CTkImage(light_image = Image.open("ninatired.png"), dark_image=Image.open("ninatired.png"), size=(100,100))
        self.image_label = customtkinter.CTkLabel(self, image=self.image,text="")
        self.image_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.combobox = customtkinter.CTkComboBox(self.frame, values=["Small Test Dataset", "Large Test Dataset", "Custom Small", "Custom Large"], command=self.set_dataset)
        self.combobox.grid(row=13, column=1, padx=10, pady=10, sticky="nsew")

        self.input_k = customtkinter.CTkEntry(self.frame, placeholder_text="Enter k (number of neighbors)")
        self.input_k.grid(row=14, column=1, padx=10, pady=10, sticky="nsew")

        self.button_forward = customtkinter.CTkButton(self.frame, text="Forward Search", fg_color="#5865F2", command=self.run_forward_search, font=("Helvetica", 14))
        self.button_forward.grid(row=15, column=1, padx=10, pady=10, sticky="nsew", columnspan=1)

        self.button_backward = customtkinter.CTkButton(self.frame, text="Backward Search", fg_color="#5865F2", command=self.run_backward_search, font=("Helvetica", 14))
        self.button_backward.grid(row=16, column=1, padx=10, pady=10, sticky="nsew", columnspan=1)

        self.dataset_choice = None
        
    def set_dataset(self, choice):
        self.dataset_choice = choice
            
    def run_forward_search(self):
        self.run_search(forward_search)

    def run_backward_search(self):
        self.run_search(backward_search)

    def run_search(self, search_function):
        print("====================Welcome to Komay and friends' Feature Search Selection!====================")
        print("Featuring:\n\n\tAdithya Iyer (aiyer026)\n\tAndy Jarean (ajare002)\n\tKomay Sugiyama (ksugi014)\n\tTingxuan Wu (twu148)\n")

        global data, features, k

        if self.dataset_choice is None:
            print("Please select a dataset.")
            return

        file = ""
        if self.dataset_choice == "Small Test Dataset":
            file = "data/small-test-dataset.txt"
        elif self.dataset_choice == "Large Test Dataset":
            file = "data/large-test-dataset.txt"
        elif self.dataset_choice == "Custom Small":
            file = "data/CS170_Spring_2024_Small_data__21.txt"
        elif self.dataset_choice == "Custom Large":
            file = "data/CS170_Spring_2024_Large_data__21.txt"

        try:
            data = pd.read_csv(file, sep="\\s+", header=None)
        except FileNotFoundError:
            print("Invalid File Name!")
            return

        num_instances = data.shape[0]
        num_features = data.shape[1]
        features = set(range(1, num_features))
        print(f"This dataset has {num_instances} instances and {num_features - 1} features (excluding class label).\n")

        print("Normalizing data using Z-Score...\n")
        normalize(data)

        try:
            k = int(self.input_k.get())
        except ValueError:
            print("Please enter a valid number for k.")
            return

        print("=====RUNNING SEARCH=====")
        s = time.time()
        f_set = search_function()
        e = time.time()
        print(f"\nFinished search in {e - s} seconds! Best Feature Set: {f_set}")

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.configure(state='normal') 
        self.text_widget.insert(customtkinter.END, s)
        self.text_widget.configure(state='disabled')  
        self.text_widget.see(customtkinter.END)  

    def flush(self):
        pass

#-----------------------------
#Functionality
#-----------------------------

# Training data
k = 1

# Load data from user input
def acquire_data(file):
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
    
    def clone_after_adding(self, feature):
        new_features = self.features.copy()
        new_features.add(feature)
        return Node(new_features, self.score)

    def clone_after_removing(self, feature):
        new_features = self.features.copy()
        new_features.discard(feature)
        return Node(new_features, self.score)

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
                # Find the indices of the k smallest distances
        k_indices = np.argpartition(distances, k)[:k]
        
        # Get the labels of the k nearest neighbors
        k_labels = self.training_data.iloc[k_indices, 0]
        
        # Determine the most common label among the k nearest neighbors
        most_common_label = Counter(k_labels).most_common(1)[0][0]
        # Return predicted label and index of closest neighbor
        return most_common_label, k_indices, distances[k_indices]

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

def evaluation_function(n, v):
    v.feature_set = n.features
    return 100*v.validate()

def expand_forward(n):
    return [n.clone_after_adding(f) for f in features if f not in n.features]
    # expanded = []
    # for f in allFeatures:
    #     if f not in n.features:
    #        expanded.append(n.cloneAfterAdding(f))
    # return expanded

def expand_backward(n):
    return [n.clone_after_removing(f) for f in features if f in n.features]
    # expanded = []
    # for f in allFeatures:
    #     if f in n.features:
    #         expanded.append(n.cloneAfterRemoving(f))
    # return expanded

def forward_search():
    node = Node()
    c = Classifier()
    validator = Validator(node.features, data, c)
    node.score = evaluation_function(node, validator)
    best = node
    print("Beginning Search...")
    while node.features != features:
        if node.score <= best.score: print("\nWarning! Accuracy Decreased!")
        else: best = node
        print(f"\nBest: {node}\n")
        neighbors = expand_forward(node)
        for n in neighbors:
            n.score = evaluation_function(n, validator)
            print(f"    {n}")
        node = max(neighbors, key=lambda n: n.score)
    return best if node.score <= best.score else node

def backward_search():
    node = Node(features)
    c = Classifier()
    validator = Validator(node.features, data, c)
    node.score = evaluation_function(node, validator)
    best = node
    print("Beginning Search...")
    while len(node.features):
        if node.score < best.score:
            print("\nWarning! Accuracy Decreased!")
        else:
            best = node
        node = max(expand_backward(node), key=lambda n: n.score)
    return best if node.score < best.score else node

if __name__ == "__main__":
    app = App()
    app.mainloop()