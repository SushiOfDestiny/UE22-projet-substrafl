# Using TensorFlow FedAvg on Colorectal Histology dataset
# %matplotlib inline
import codecs
import os
import sys
import zipfile
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_algorithms.weight_manager as weight_manager

# import tf.keras
# Setup
# *****

# This examples runs with three organizations. Two organizations provide datasets, while a third
# one provides the algorithm.

# In the activation following code cell, we define the different organizations needed for our FL experiment.


from substra import Client

N_CLIENTS = 3

# Every computation will run in `subprocess` mode, where everything runs locally in Python
# subprocesses.
# Ohers backend_types are:
# "docker" mode where computations run locally in docker containers
# "remote" mode where computations run remotely (you need to have a deployed platform for that)
client_0 = Client(backend_type="subprocess")
client_1 = Client(backend_type="subprocess")
client_2 = Client(backend_type="subprocess")
# To run in remote mode you have to also use the function `Client.login(username, password)`

clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
}


# Store organization IDs
ORGS_ID = list(clients.keys())
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data providers orgs are the two last organizations.
# Data Preparation

# This section downloads (if needed) the **histology dataset**.
from tf_fedavg_assets.dataset.histo_dataset import setup_histo

# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)
data_path = pathlib.Path.cwd() / "tmp" / "data_histo"

setup_histo(data_path, len(DATA_PROVIDER_ORGS_ID))
# Visualizing the Dataset
# visualize dataset
images = np.load('tmp/data_histo/org_1/train/train_images.npy')
labels = np.load('tmp/data_histo/org_1/train/train_labels.npy')

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(images[i])
#     plt.xlabel(labels[i])
# plt.show()
    
min(labels), max(labels) # labels go from 0 to 7
# Dataset registration

# A `documentation/concepts:Dataset` is composed of an **opener**, which is a Python script that can load
# the data from the files in memory and a description markdown file.
# The `documentation/concepts:Dataset` object itself does not contain the data. The proper asset that contains the
# data is the **datasample asset**.

# A **datasample** contains a local path to the data. A datasample can be linked to a dataset in order to add data to a
# dataset.

# Data privacy is a key concept for Federated Learning experiments. That is why we set
# `documentation/concepts:Permissions` for `documentation/concepts:Assets` to determine how each organization can access a specific asset.

# Note that metadata such as the assets' creation date and the asset owner are visible to all the organizations of a
# network.
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = pathlib.Path.cwd() / "tf_fedavg_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):

    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="histo",
        type="npy",
        data_opener=assets_directory / "dataset" / "histo_opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing dataset key"

    # Add the training data on each organization.
    # now data_sample will have keys like "images" and "labels"
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(data_sample)
# # Metric registration

# A metric is a function used to evaluate the performance of your model on one or several
# **datasamples**.

# To add a metric, you need to define a function that computes and returns a performance
# from the datasamples (as returned by the opener) and the predictions_path (to be loaded within the function).

# When using a TF SubstraFL algorithm, the predictions are saved in the `predict` function in numpy format
# so that you can simply load them using `np.load`.

# After defining the metrics, dependencies, and permissions, we use the `add_metric` function to register the metric.
# This metric will be used on the test datasamples to evaluate the model performances.
from sklearn.metrics import accuracy_score
import numpy as np

from substrafl.dependency import Dependency
from substrafl.remote.register import add_metric

permissions_metric = Permissions(public=False, authorized_ids=[ALGO_ORG_ID] + DATA_PROVIDER_ORGS_ID)

# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
metric_deps = Dependency(pypi_dependencies=["numpy==1.23.1", "scikit-learn==1.1.1"])


def accuracy(datasamples, predictions_path):
    y_true = datasamples["labels"]
    y_pred = np.load(predictions_path)

    return accuracy_score(y_true, np.argmax(y_pred, axis=1))


metric_key = add_metric(
    client=clients[ALGO_ORG_ID],
    metric_function=accuracy,
    permissions=permissions_metric,
    dependencies=metric_deps,
)
# Specify the machine learning components
# ***************************************
# This section uses the future TF based SubstraFL API to simplify the definition of machine learning components.
# However, SubstraFL is compatible with any machine learning framework.


# In this section, you will:

# - Register a model and its dependencies
# - Specify the federated learning strategy
# - Specify the training and aggregation nodes
# - Specify the test nodes
# - Actually run the computations
# # Model definition

# We choose to use a classic torch CNN as the model to train. The model structure is defined by the user independently
# of SubstraFL.
seed = 42
tf.random.set_seed(seed)

# @tf.keras.utils.register_keras_serializable(tf.keras.Sequential)
class CNN(tf.keras.Sequential):
    def __init__(self):
        super(CNN, self).__init__(layers=[
            tf.keras.layers.Rescaling(1./255, input_shape=(150,150,3)),
            tf.keras.layers.Conv2D(32, 5, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 5, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 5, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(8, activation= 'softmax')
        ])

model = CNN()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Add model compiling
model.compile(optimizer=optimizer, loss=criterion)

# no scheduler used in the example

####################
# SERIALIZING   
# with dict
model_state_dict = weight_manager.model_state_dict(model)
# optimizer_state_dict = optimizer.get_config()


# Specifying on how much data to train

# To specify on how much data to train at each round, we use the `index_generator` object.
# We specify the batch size and the number of batches to consider for each round (called `num_updates`).
# See `substrafl_doc/substrafl_overview:Index Generator` for more details.

from substrafl.index_generator import NpIndexGenerator

# Number of model updates between each FL strategy aggregation.
NUM_UPDATES = 100

# Number of samples per update.
BATCH_SIZE = 32

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)
# TensorFlow Dataset definition

# This tf Dataset is used to preprocess the data using the `__getitem__` function.

# This tf Dataset needs to have a specific `__init__` signature, that must contain (self, datasamples, is_inference).

# The `__getitem__` function is expected to return (inputs, outputs) if `is_inference` is `False`, else only the inputs.
# This behavior can be changed by re-writing the `_local_train` or `predict` methods.


# img = images[0][None,...]
# img2 = img / 255
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(img[0])
# plt.subplot(1,2,2)
# plt.imshow(img2[0])
# plt.show()

# images[0][None,...].shape
# images[0][...].shape
# images[0].shape
# images2 = np.copy(images)
# images2 = images2 / 255.
# images
# images2
# images[0].dtype, images2[0].dtype
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(images[0])
# plt.subplot(1,2,2)
# plt.imshow(images2[0])
# plt.show()


class TFDataset(tf.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["images"] / 255. # new datasamples with normalized datas
        self.y = datasamples["labels"]
        self.is_inference = is_inference
        self.nb_classes = 8 # labels go from 0 to 7
        self.one_hots = tf.one_hot(indices=list(range(self.nb_classes)), depth=self.nb_classes, dtype='float32')
        # =
        # [[1., 0., 0., 0., 0., 0., 0., 0.],
        #  [0., 1., 0., 0., 0., 0., 0., 0.],
        #  [0., 0., 1., 0., 0., 0., 0., 0.],
        #  ...,
        # ]
        self.input_shape = self.x[0].shape # shape of input images (150,150,3)


    def __getitem__(self, idx):

        if self.is_inference:
            x = tf.convert_to_tensor(value=self.x[idx][None, ...], dtype='float64') # keep float64
            return x

        else:
            x = tf.convert_to_tensor(value=self.x[idx][None, ...], dtype='float64')
            y = self.one_hots[self.y[idx]]
            return x, y

    def __len__(self):
        return len(self.x)
    

    # adding missing methods

    def _inputs(self):
        """Returns a list of the input datasets of the dataset."""
        if self.is_inference:
            return []
        else:
            return tf.TensorSpec(shape=self.input_shape, dtype=tf.float64), tf.TensorSpec(shape=(self.nb_classes,), dtype=tf.float32)

    def element_spec(self):
        """The type specification of an element of this dataset."""
        if self.is_inference:
            return tf.TensorSpec(shape=self.input_shape, dtype=tf.float64)
        else:
            return tf.TensorSpec(shape=self.input_shape, dtype=tf.float64), tf.TensorSpec(shape=(self.nb_classes,), dtype=tf.float32)
# to test the previous class, we recreate a data_sample
# we assume a datasample is like
# dict('images': np.array,
#      'labels': np.array)
data_sample1 = dict(
    {
    'images': images[:10],
    'labels': labels[:10]
    }
)
data_sample1['images'][0].shape
test_ds = TFDataset(datasamples=data_sample1, is_inference=False)
# Test of tf_data__loader
# data = tf.data.Dataset.from_tensor_slices(test_ds)
# fail so tf_data_loader doesn't work atm
# SubstraFL algo definition
from tensorflow_algorithms.tf_fed_avg_algo import TFFedAvgAlgo


class MyAlgo(TFFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model_state_dict,
            criterion=criterion,
            optimizer=None,
            index_generator=index_generator,
            dataset=TFDataset,
            seed=seed,
        )



from substrafl.strategies import FedAvg

strategy = FedAvg()


from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


aggregation_node = AggregationNode(ALGO_ORG_ID)

train_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:

    # Create the Train Data Node (or training task) and save it in a list
    train_data_node = TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    train_data_nodes.append(train_data_node)

from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy


test_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:

    # Create the Test Data Node (or testing task) and save it in a list
    test_data_node = TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        test_data_sample_keys=[test_datasample_keys[org_id]],
        metric_keys=[metric_key],
    )
    test_data_nodes.append(test_data_node)

# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=1)


from substrafl.experiment import execute_experiment

# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = 3

# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
algo_deps = Dependency(pypi_dependencies=["numpy==1.23.1", "tensorflow=2.12.0"], local_code=[pathlib.Path.cwd() / "tensorflow_algorithms"])

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    algo=MyAlgo(),
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=algo_deps,
)