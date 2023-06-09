import abc
import inspect
import logging
import os
import shutil
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf 

from substrafl.algorithms.algo import Algo
from substrafl.exceptions import BatchSizeNotFoundError
from substrafl.exceptions import DatasetSignatureError
from substrafl.exceptions import DatasetTypeError
from substrafl.exceptions import OptimizerValueError
from substrafl.index_generator import BaseIndexGenerator
from substrafl.remote.decorators import remote_data

from tf_data_loader import tf_dataloader
import weight_manager

import pickle


logger = logging.getLogger(__name__)

class TFAlgo(Algo):

    def __init__(
        self,
        model: tf.keras.Sequential,
        criterion: tf.keras.losses.Loss,
        index_generator: Union[BaseIndexGenerator, None],
        dataset: tf.data.Dataset,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        """The ``__init__`` functions is called at each call of the `train()` or `predict()` function
        For round>2, some attributes will then be overwritten by their previous states in the `load()` function,
        before the `train()` or `predict()` function is ran.
        """
        super().__init__(*args, **kwargs)

        if seed is not None:
            tf.random.set_seed(seed)

        self._device = self._get_tf_device(use_gpu=use_gpu)

        with tf.device(self._device):
            self._optimizer = optimizer
            # Move the optimizer to GPU if needed
            #https://www.tensorflow.org/guide/gpu
            if self._optimizer is not None:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Restrict TensorFlow to only use the first GPU
                    try:
                        tf.config.set_visible_devices(gpus, 'GPU')
                        logical_gpus = tf.config.list_logical_devices('GPU')
                    except RuntimeError as e:
                        # Visible devices must be set before GPUs have been initialized
                        print(e)
            self._criterion = criterion
            self._scheduler = scheduler

            self._index_generator = index_generator
            self._dataset: tf.data.Dataset = dataset
            # dataset check overlooked

    @property
    def model(self) -> tf.keras.Sequential:
        """Model exposed when the user downloads the model

        Returns:
            torch.nn.Module: model
        """
        return self._model

    @abc.abstractmethod
    def train(
        self,
        datasamples: Any,
        shared_state: Any = None,
    ) -> Any:
        # Must be implemented in the child class
        raise NotImplementedError()
    
    @remote_data
    def predict(self, datasamples: Any, shared_state: Any = None, predictions_path: os.PathLike = None) -> Any:
        """Execute the following operations:

            * Create the test torch dataset.
            * Execute and return the results of the ``self._local_predict`` method

        Args:
            datasamples (typing.Any): Input data
            shared_state (typing.Any): Latest train task shared state (output of the train method)
            predictions_path (os.PathLike): Destination file to save predictions
        """

        # Create tf dataset
        predict_dataset = self._dataset(datasamples, is_inference=True)
        self._local_predict(predict_dataset=predict_dataset, predictions_path=predictions_path)
    
    def _save_predictions(self, predictions: tf.Tensor, predictions_path: os.PathLike):
        """Save the predictions under the numpy format.

        Args:
            predictions (tf.Tensor): predictions to save.
            predictions_path (os.PathLike): destination file to save predictions.
        """
        if predictions_path is not None:
            np.save(predictions_path, predictions)
            # Create a folder ??
            shutil.move(str(predictions_path) + ".npy", predictions_path)
    
    def _local_predict(self, predict_dataset: tf.data.Dataset, predictions_path):
        """Execute the following operations:

            * Create the torch dataloader using the index generator batch size.
            * Set the model to `eval` mode
            * Save the predictions using the
              :py:func:`~substrafl.algorithms.tensorflow.tf_base_algo.TFAlgo._save_predictions` function.

        Args:
            predict_dataset (tf.data.Dataset): predict_dataset build from the x returned by the opener.

        Important:
            The onus is on the user to ``save`` the compute predictions. Substrafl provides the
            :py:func:`~substrafl.algorithms.tensorflow.tf_base_algo.TFAlgo._save_predictions` to do so.
            The user can load those predictions from a metric file with the command:
            ``y_pred = np.load(inputs['predictions'])``.

        Raises:
            BatchSizeNotFoundError: No default batch size have been found to perform local prediction.
                Please overwrite the predict function of your algorithm.
        """
        if self._index_generator is not None:
            # predict_loader = tf_dataloader(predict_dataset, batch_size=self._index_generator.batch_size)
            predict_loader = predict_dataset
        else:
            raise BatchSizeNotFoundError(
                "No default batch size has been found to perform local prediction. "
                "Please overwrite the _local_predict function of your algorithm."
            )

        # Equivalent of self._model.eval() : desactivate the variables not used for prediction
        tf.keras.backend.set_learning_phase(0)
        # Variable controlling the inference mode
        inference_mode = tf.Variable(True, trainable=False)

        predictions = tf.constant([])
        if inference_mode:
            # Code specific to the inference mode
            with tf.device(self._device):
                for x in predict_loader:
                    predictions = tf.concat([predictions, self._model(x)], 0)

        with tf.device('CPU:0'):
            # https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
            predictions = tf.identity(predictions)
            self._save_predictions(predictions, predictions_path)

    def _local_train(
            self,
            train_dataset: tf.data.Dataset,
        ):
            """Local train method. Contains the local training loop.

            Train the model on ``num_updates`` minibatches, using the ``self._index_generator generator`` as batch sampler
            for the tf dataset.

            Args:
                train_dataset (TFDataset / tf.data.Dataset): train_dataset build from the x and y returned by the opener.

            Important:

                You must use ``next(self._index_generator)`` as batch sampler,
                to ensure that the batches you are using are correct between 2 rounds
                of the federated learning strategy.
            """
            if self._optimizer is None:
                raise OptimizerValueError(
                    "No optimizer found. Either give one or overwrite the _local_train method from the used torch"
                    "algorithm."
                )

            # Create tf dataloader
            # train_data_loader = tf_dataloader(train_dataset, batch_sampler=self._index_generator) # reminder: class(train_dataset) = TFDataset / tf.data.Dataset
            # avoiding the loader probleme
            train_data_loader = train_dataset

            # Changing device
            with tf.device(self._device):
                for x_batch, y_batch in train_data_loader:
                    
                    # x_batch = x_batch.to(self._device)
                    # y_batch = y_batch.to(self._device)

                    # cf https://www.tensorflow.org/overview?hl=fr "For experts"
                    # Forward pass
                    y_pred = self._model(x_batch, training=True)

                    # Compute loss
                    loss = self._criterion(y_batch, y_pred)

                    # Calculate gradients
                    grads = tf.GradientTape.gradient(loss, self._model.trainable_variables)

                    # Apply gradients
                    self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

                    if self._scheduler is not None:
                        self._scheduler.step()


    def _update_from_checkpoint(self, path: Path) -> dict:
        """Load the checkpoint and update the internal state
        from it.
        Pop the values from the checkpoint so that we can ensure that it is empty at the
        end, i.e. all the values have been used.

        Args:
            path (pathlib.Path): path where the checkpoint is saved

        Returns:
            dict: checkpoint

        Example:

            .. code-block:: python

                def _update_from_checkpoint(self, path: Path) -> dict:
                    checkpoint = super()._update_from_checkpoint(path=path)
                    self._strategy_specific_variable = checkpoint.pop("strategy_specific_variable")
                    return checkpoint
        """
        assert path.is_file(), f'Cannot load the model - does not exist {list(path.parent.glob("*"))}'
        
        # we ignore the map_location arg
        with open(path, "rb") as f:
            checkpoint = pickle.load(path)
        
        weight_manager.model_load_state_dict(self._model, checkpoint.pop("model_state_dict"))

        if self._optimizer is not None:
            self._optimizer.from_config(checkpoint.pop("optimizer_state_dict"))

        if self._scheduler is not None:
            self._scheduler.from_config(checkpoint.pop("scheduler_state_dict"))


        self._index_generator = checkpoint.pop("index_generator")

        # following Torch code has not been implemented
        # if self._device == tf.device("cpu"):
        #     torch.set_rng_state(checkpoint.pop("rng_state").to(self._device))
        # else:
        #     torch.cuda.set_rng_state(checkpoint.pop("rng_state").to("cpu"))

        return checkpoint
    
    def load(self, path: Path) -> "TFAlgo":
        """Load the stateful arguments of this class.
        Child classes do not need to override that function.

        Args:
            path (pathlib.Path): The path where the class has been saved.

        Returns:
            TorchAlgo: The class with the loaded elements.
        """
        checkpoint = self._update_from_checkpoint(path=path)
        assert len(checkpoint) == 0, f"Not all values from the checkpoint have been used: {checkpoint.keys()}"
        return self
    
    def _get_state_to_save(self) -> dict:
        """Create the algo checkpoint: a dictionary
        saved with tf ???.
        In this algo, it contains the state to save for every strategy.
        Reimplement in the child class to add strategy-specific variables.

        Example:

            .. code-block:: python

                def _get_state_to_save(self) -> dict:
                    local_state = super()._get_state_to_save()
                    local_state.update({
                        "strategy_specific_variable": self._strategy_specific_variable,
                    })
                    return local_state

        Returns:
            dict: checkpoint to save
        """
        checkpoint = {
            "model_state_dict": weight_manager.model_state_dict(self._model),
            "index_generator": self._index_generator,
        }
        if self._optimizer is not None:
            # for an tf.optimizers.Optimizer, we use .get_config() and .from_config()
            checkpoint["optimizer_state_dict"] = self._optimizer.get_config()
        if self._scheduler is not None:
            # for an tf.optimizers.Optimizer, we use .get_config() and .from_config()
            checkpoint["scheduler_state_dict"] = self._scheduler.get_config()

        return checkpoint
    
    def _check_tf_dataset(self):
        # Check that the given Dataset is not an instance
        try:
            issubclass(self._dataset, tf.data.Dataset)
        except TypeError:
            raise DatasetTypeError(
                "``dataset`` should be non-instantiate tf.data.Dataset class. "
                "This means that calling ``dataset(datasamples, is_inference=False)`` must "
                "returns a tf dataset object. "
                "You might have provided an instantiate dataset or an object of the wrong type."
            )

        # Check the signature of the __init__() function of the tf dataset class
        signature = inspect.signature(self._dataset.__init__)
        init_parameters = signature.parameters

        if "datasamples" not in init_parameters:
            raise DatasetSignatureError(
                "The __init__() function of the tf Dataset must contain datasamples as parameter."
            )
        elif "is_inference" not in init_parameters:
            raise DatasetSignatureError(
                "The __init__() function of the tf Dataset must contain is_inference as parameter."
            )
        
    def save(self, path: Path):
        """Saves all the stateful elements of the class to the specified path.
        Child classes do not need to override that function.

        Args:
            path (pathlib.Path): A path where to save the class.
        """
        
        # tf functions to save and load objecrs are quite different than torch's, we firstly try to imitate the latter 
        # with the pickle module
        with open(path, "wb") as f:
            pickle.dump(self._get_state_to_save(), f)

        assert path.is_file(), f'Did not save the model properly {list(path.parent.glob("*"))}'
    
    def summary(self):
        """Summary of the class to be exposed in the experiment summary file.
        Implement this function in the child class to add strategy-specific variables. The variables
        must be json-serializable.

        Example:

            .. code-block:: python

                def summary(self):
                    summary = super().summary()
                    summary.update(
                        "strategy_specific_variable": self._strategy_specific_variable,
                    )
                    return summary

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        summary.update(
            {
                "model": str(type(self._model)),
                "criterion": str(type(self._criterion)),
                "optimizer": None
                if self._optimizer is None
                else {
                    "type": str(type(self._optimizer)),
                    "parameters": self._optimizer.defaults,
                },
                "scheduler": None if self._scheduler is None else str(type(self._scheduler)),
            }
        )
        return summary

