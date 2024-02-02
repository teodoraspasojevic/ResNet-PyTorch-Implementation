import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

data = pd.read_csv('data.csv', sep=';')
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

train_dataset = ChallengeDataset(data=train_df, mode='train')
test_dataset = ChallengeDataset(data=test_df, mode='val')

# train_dataset.oversample_unbalanced_classes()
# test_dataset.oversample_unbalanced_classes()

# Generate new oversampled data
# new_data = [train_dataset[idx][0] for idx in oversample_indices]  # Get data
# new_labels = [train_dataset[idx][1] for idx in oversample_indices]  # Get labels

# Create the DataLoaders for training and validation
# Set the batch_size to the desired number of images per batch
# Set shuffle=True for the training data loader to shuffle the dataset before creating batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# create an instance of our ResNet model

resnet_model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion

loss = t.nn.BCELoss()
optim = t.optim.Adam(resnet_model.parameters(), lr=0.001)

trainer = Trainer(model=resnet_model, crit=loss, optim=optim, train_dl=train_loader, val_test_dl=val_loader, cuda=True, early_stopping_patience=5)

# go, go, go... call fit on trainer

res = trainer.fit(epochs=200)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
