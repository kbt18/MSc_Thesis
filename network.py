from pyNN.utility import get_simulator, init_logging, normalized_filename
import numpy as np
import torch
import sys
import os
import modelling_utils as utils
import torchvision
import matplotlib.pyplot as plt


# === Configure the simulator ================================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=1.0, min_delay=1.0) # timestep=1.0 (default)

time_per_image = 150
nb_images = 5 # max 10000
fashion_mnist = False

# === Load the dataset =========================================================
if (fashion_mnist):
    root = os.path.expanduser("~/data/datasets/torch/fashion-mnist")
    test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)
else:
    root = os.path.expanduser("~/data/datasets/torch/mnist")
    test_dataset = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform=None, download=True)

x_test = np.array(test_dataset.data, dtype=np.float)
y_test  = np.array(test_dataset.targets, dtype=np.int)

spike_times = utils.stack_spikes(x_test, offset=time_per_image)

# === Create the neural populations ============================================
inf = sys.float_info.max
nb_input = 28*28
nb_hidden = 100
nb_output = 10

cell_parameters = {
        'v_rest':     0.0,  # Resting membrane potential in mV.
        'v_reset':    0.0,  # Reset potential after a spike in mV.
        'v_thresh':   1.0,  # Spike threshold in mV.
        'cm':         1.0,  # Capacity of the membrane in nF
        'tau_m':      10.0, # Membrane time constant in ms.
        'tau_refrac': 1.0,  # Duration of refractory period in ms. (CANT BE LESS THAN TIMESTEP, CAUSES BUG)
        'tau_syn_E':  5.0, # Decay/rise time of excitatory synaptic current in ms.
        'tau_syn_I':  5.0, # Decay/rise time of inhibitory synaptic current in ms.
        'i_offset':   0.0,  # Offset current in nA
}

output_cell_parameters = cell_parameters.copy()
output_cell_parameters['v_thresh'] = inf
output_cell_parameters['cm'] = 1.0

input = sim.Population(nb_input, sim.SpikeSourceArray(spike_times=spike_times), label="input")
hidden = sim.Population(nb_hidden, sim.IF_curr_exp(**cell_parameters), label="hidden")
output = sim.Population(nb_output, sim.IF_curr_exp(**output_cell_parameters), label="output")

w1 = utils.pt_to_np('weights/MNIST_w1.pt')
w2 = utils.pt_to_np('weights/MNIST_w2.pt')

connections_input = utils.connections_all_to_all(nb_input, nb_hidden, w1)
connections_hidden = utils.connections_all_to_all(nb_hidden, nb_output, w2)

input_to_hidden = sim.Projection(input, hidden, sim.FromListConnector(connections_input, column_names=['weight']))
hidden_to_output = sim.Projection(hidden, output, sim.FromListConnector(connections_hidden, column_names=['weight']))

input.record("spikes")
hidden.record(["spikes", "v"])
output.record("v")

# === Run the simulation =====================================================
dataset_size = np.shape(x_test)[0]

print('starting simulation')

sim.run(time_per_image * nb_images)

print('stopping simulation')

# === Save the results, optionally plot a figure ===============================

if (fashion_mnist):
    save_dir = "ResultsFashionMNIST"
else:
    save_dir = "ResultsMNIST"

filename = normalized_filename(save_dir, "input", "pkl", options.simulator)
input.write_data(filename, annotations={'script_name': __file__})

filename = normalized_filename(save_dir, "hidden", "pkl", options.simulator)
hidden.write_data(filename, annotations={'script_name': __file__})

filename = normalized_filename(save_dir, "output", "pkl", options.simulator)
output.write_data(filename, annotations={'script_name': __file__})

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(hidden.get_data().segments[0].spiketrains,
              ylabel="Membrane potential (mV)",
              yticks=True, ylim=(-66, -48)),
        Panel(hidden.get_data().segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              yticks=True, ylim=(-1.2, 1.2), legend=False),
        Panel(output.get_data().segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[output.label], yticks=True, ylim=(-3, 3)),
        # title="Membrane potential of hidden and output layers",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
