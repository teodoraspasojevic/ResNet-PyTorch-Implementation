import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

data = pd.read_csv('data.csv', sep=';')
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

train_dataset = ChallengeDataset(data=train_df, mode='train')
test_dataset = ChallengeDataset(data=test_df, mode='val')

# create an instance of our ResNet model

resnet_model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion

loss = t.nn.BCELoss()
optim = t.optim.SGD(resnet_model.parameters(), lr=0.01)

trainer = Trainer(model=resnet_model, crit=loss, optim=optim, train_dl=train_dataset, val_test_dl=test_dataset, cuda=True)

# go, go, go... call fit on trainer

res = trainer.fit(epochs=2)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
