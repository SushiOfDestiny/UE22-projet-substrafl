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

        self._model = model.to(self._device)
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
        predict_dataset = self._dataset(datasamples)
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
            predict_loader = tf_dataloader(predict_dataset, batch_size=self._index_generator.batch_size)
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

        with tf.device('GPU:0'):
            # https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
            predictions = tf.identity(predictions)
            self._save_predictions(predictions, predictions_path)

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
            checkpoint["optimizer_state_dict"] = self._optimizer.state_dict()


        if self._device == torch.device("cpu"):
            checkpoint["rng_state"] = torch.get_rng_state()
        else:
            checkpoint["rng_state"] = torch.cuda.get_rng_state()

        return checkpoint