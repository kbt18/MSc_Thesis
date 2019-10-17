import numpy as np
import pickle
import sys
import torchvision
import os
from pyNN.utility.plotting import Figure, Panel
import modelling_utils as utils

#-------------------------------------------------------------------------------
nb_images = 10000
time_per_image = 150
n_classes = 10
latency = 15 # milliseconds
fashion_mnist = False

# locations of recorded network activity
HIDDEN_DATA = ""
INPUT_DATA = ""
OUTPUT_DATA = ""

# Load the data
dirname = os.path.dirname(__file__)
hidden_filename = os.path.join(dirname, HIDDEN_DATA)
input_filename = os.path.join(dirname, INPUT_DATA)
output_filename = os.path.join(dirname, OUTPUT_DATA)

hidden_activity = np.load(hidden_filename, allow_pickle=True)
input_activity = np.load(input_filename, allow_pickle=True)
output_activity = np.load(output_filename, allow_pickle=True)

input_spikes = input_activity.segments[0].spiketrains
hidden_spikes = hidden_activity.segments[0].spiketrains
output_potentials = output_activity.segments[0].filter(name='v')[0]
output_potentials = np.array(output_potentials)

print("average spikes")
total_spikes = utils.count_spikes(input_spikes) + utils.count_spikes(hidden_spikes)
print(total_spikes/nb_images)

# Remove first row
output_potentials = np.delete(output_potentials, 0, 0)
first_result = output_potentials[time_per_image:time_per_image*2, :]

# Reshape dataset
output_potentials = np.reshape(output_potentials, (nb_images, time_per_image, n_classes), order='A')
first = output_potentials[1]

# Get classes
classes = np.zeros(nb_images)

for i in range(nb_images):
    classes[i] = np.asscalar((np.where(output_potentials[i, :latency] == np.amax(output_potentials[i, :latency]))[1])[0])

# Get accuracy
if (fashion_mnist):
    root = os.path.expanduser("~/data/datasets/torch/fashion-mnist")
    test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)
else:
    root = os.path.expanduser("~/data/datasets/torch/mnist")
    test_dataset = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform=None, download=True)

y_test  = np.array(test_dataset.targets, dtype=np.int)[:nb_images]

if (n_classes==2):
    y_test  = np.array(test_dataset.targets, dtype=np.int)
    indexes_to_remove_test = np.argwhere(y_test > 1).flatten()
    y_test = np.delete(y_test, indexes_to_remove_test)

print("accuracy: " + str(np.average(classes.astype(int)==y_test)))
