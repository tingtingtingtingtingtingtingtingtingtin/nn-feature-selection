import numpy as np
import struct


# Function to load training data
def load_training_data(file_path):
    data = []
    with open(file_path, 'rb') as file:
        while True:
            bytes_read = file.read(8)  # IEEE 754 double precision is 8 bytes
            if not bytes_read:
                break
            # Convert the bytes to a float using struct
            number = struct.unpack('d', bytes_read)[0]
            data.append(number)
    
    # Convert the list to a NumPy array for easier manipulation
    data_array = np.array(data)
    return data_array

# Testing

small = "data/small-test-dataset.txt"
large = "data/large-test-dataset.txt"

smallDataset = load_training_data(small)
largeDataset = load_training_data(large)

print(smallDataset)
print(largeDataset)