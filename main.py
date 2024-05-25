import copy
import queue
import random
import tkinter 
from tkinter import scrolledtext
import customtkinter
import sys

#-----------------------------
#UI
#-----------------------------
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class TextFrame(customtkinter.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.label = customtkinter.CTkLabel(master=self, text=title)
        self.label.grid(row=0, column=0, sticky="nsew", pady=(10, 0))
        self.text_widget = customtkinter.CTkTextbox(master=self, width=400, height=600, corner_radius=0)
        self.text_widget.grid(row=1, column=0, sticky="nsew")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("NN Feature Selection")
        self.geometry("500x500")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.text_frame = TextFrame(self, title="NN Feature Selection")
        self.text_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")

        sys.stdout = TextRedirector(self.text_frame.text_widget)

        self.button = customtkinter.CTkButton(self, text="forward search", command=forward_search)
        self.button.grid(row=3, column=0, padx=10, pady=10, sticky="ew", columnspan=2)
        self.button = customtkinter.CTkButton(self, text="backward search", command=backward_search)
        self.button.grid(row=4, column=0, padx=10, pady=10, sticky="ew", columnspan=2)

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.configure(state='normal')  # Make it writable
        self.text_widget.insert(customtkinter.END, s)
        self.text_widget.configure(state='disabled')  # Make it read-only again
        self.text_widget.see(customtkinter.END)  # Scroll to the end

    def flush(self):
        pass


#ABOVE IS THE EXAMPLE FROM THE DOCUMENTATION. NOT MY CODE, I WAS JUST PLAYING AROUND TO SEE WHAT IT LOOKS LIKE



#-----------------------------
#Functionality
#-----------------------------

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

print("=====FORWARD SEARCH=====")
print(f"\nFinished Search! Best Feature Set: {forward_search()}")
print()
print("=====BACKWARD SEARCH=====")
print(f"\nFinished Search! Best Feature Set: {backward_search()}")


if __name__ == "__main__":
    app = App()
    app.mainloop()