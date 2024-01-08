import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
'''
Quick script to visually inspect the data made by make_datset.py, in order to check, that the data generated looks correct
'''


# Load in training set
training_data = torch.load("./data/processed/train_dataset.pt")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
print(f"Label: {label}")
plt.show()


### Same for the test
test_data = torch.load("./data/processed/test_dataset.pt")

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Display image and label.
train_features, train_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
print(f"Label: {label}")
plt.show()