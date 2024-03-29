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

from tensorflow_algorithms.tf_data_loader import tf_dataloader
import tensorflow_algorithms.weight_manager as weight_manager

import cloudpickle


logger = logging.getLogger(__name__)


class TFAlgo(Algo):
    def __init__(
        self,
        model: dict,
        criterion: tf.keras.losses.Loss,
        index_generator: Union[BaseIndexGenerator, None],
        dataset: tf.data.Dataset,
        optimizer: Optional[dict] = None,
        scheduler: Optional[dict] = None,
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

        # we receive here a dict of compiled model with the right config
        self._model = model

        self._optimizer = optimizer
        assert self._optimizer is None, "Optimizer is not None"
        # Optimizer information are in the model information

        self._criterion = criterion

        self._scheduler = scheduler
        assert self._scheduler is None, "Scheduler is not None"
        # No Scheduler is used

        self._index_generator = index_generator
        assert self._index_generator is None, "Index Generator is not None"
        # No Index Generator is used

        self._dataset: tf.data.Dataset = dataset
        # Dataset check overlooked, because we assumed it is correctly chosen

    @property
    def model(self) -> dict:
        """Model exposed when the user downloads the model

        Returns:
            dict: model
        """
        return self._model

    def model_deserialize(self) -> tf.keras.Sequential:
        """Return a compiled model with the particular config,
        compile config and weights contained in the self._model attribute.
        We choose a Sequential model eventhough it is a custom CNN.
        """

        new_model = weight_manager.model_load_state_dict(self._model)

        return new_model

    def model_serialize(self, model: tf.keras.Sequential) -> None:
        """Updates the self._model attribute with the state_dict (config, compile config and weights) of the model
        passed as argument.
        """

        self._model = weight_manager.model_state_dict(model)

    @abc.abstractmethod
    def train(
        self,
        datasamples: Any,
        shared_state: Any = None,
    ) -> Any:
        # Must be implemented in the child class
        raise NotImplementedError()

    @remote_data
    def predict(
        self,
        datasamples: Any,
        shared_state: Any = None,
        predictions_path: os.PathLike = None,
    ) -> Any:
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
        self._local_predict(
            predict_dataset=predict_dataset, predictions_path=predictions_path
        )

    def _save_predictions(self, predictions: tf.Tensor, predictions_path: os.PathLike):
        """Save the predictions under the numpy format.

        Args:
            predictions (tf.Tensor): predictions to save.
            predictions_path (os.PathLike): destination file to save predictions.
        """
        if predictions_path is not None:
            np.save(predictions_path, predictions)
            shutil.move(str(predictions_path) + ".npy", predictions_path)

    def _local_predict(self, predict_dataset: tf.data.Dataset, predictions_path):
        """Args:
            predict_dataset (tf.data.Dataset)

        Important:
            The onus is on the user to ``save`` the compute predictions. Substrafl provides the
            :py:func:`~substrafl.algorithms.tensorflow.tf_base_algo.TFAlgo._save_predictions` to do so.
            The user can load those predictions from a metric file with the command:
            ``y_pred = np.load(inputs['predictions'])``.
        """

        predictions = []

        # TESTS
        before_pred = self._model['weights'][2][0][0][0][:5]
        # print(f"avant prédiction (et désérialisation): {before_pred}")

        # Deserialization and compiling
        model = self.model_deserialize()

        # Normally, following code does not updates weight
        # Gathering predictions of the model
        for i in range(len(predict_dataset)):
            x = predict_dataset[i]
            y = model(x)[0]
            predictions.append(y)

        self._save_predictions(predictions, predictions_path)

        # Reserialization, eventhough the model has not theoretically changed
        self.model_serialize(model)

        # TESTS
        after_pred = self._model['weights'][2][0][0][0][:5]
        # print(f"après prédiction (et resérialisation): {after_pred}")

        print(f"Variation des poids du modèle après - avant prédiction : { after_pred - before_pred }")

    def _local_train(
        self,
        train_dataset: tf.data.Dataset,
    ):
        """Local train method. Contains the local training loop.

        Train the model on 1 epoch with a batch size of 32 (simplified setup that does not require
        any index generator)

        Args:
            train_dataset (TFDataset / tf.data.Dataset): train_dataset build from the x and y returned by the opener.
        """

        #TESTS
        before_train = self._model['weights'][2][0][0][0][:5]
        # print(f"avant entraînement (et désérialisation): {before_train}")

        # Deserialization and compiling w/ optimizer and loss
        model = self.model_deserialize()

        # Avoiding the data_loader problem
        train_data_loader = train_dataset

        # We use the simplified keras method `fit` to train the model
        model.fit(x=train_data_loader.x, y=train_data_loader.y, batch_size=32, epochs=1)

        after_train = model.get_weights()[2][0][0][0][:5]
        # print(f"après entraînement (et avant resérialisation): {after_train}")
        
        print(f"Variation des poids du modèle après - avant entrainement (avant sérialisation) : { after_train - before_train }")

        # Reserialization
        self.model_serialize(model)
        after_train_2 = self._model['weights'][2][0][0][0][:5]
        # print(f"après entraînement (et resérialisation): {after_train_2}")

        print(f"Variation des poids du modèle après - avant sérialisation : { after_train_2 - after_train }")

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
        assert (
            path.is_file()
        ), f'Cannot load the model - does not exist {list(path.parent.glob("*"))}'

        # we ignore the map_location arg
        with open(path, "rb") as f:
            checkpoint = cloudpickle.load(f)

        self._model = checkpoint.pop("model_state_dict")

        if self._optimizer is not None:
            self._optimizer = checkpoint.pop("optimizer_state_dict")

        if self._scheduler is not None:
            self._scheduler.from_config(checkpoint.pop("scheduler_state_dict"))

        self._index_generator = checkpoint.pop("index_generator")

        # following Torch code has not been implemented
        # if self._device == tf.device("cpu"):
        #     torch.set_rng_state(checkpoint.pop("rng_state").to(self._device))
        # else:
        #     torch.cuda.set_rng_state(checkpoint.pop("rng_state").to("cpu"))

        # at this point checkpoint should be empty
        return checkpoint

    def load(self, path: Path) -> "TFAlgo":
        """Load the stateful arguments of this class.
        Child classes do not need to override that function.

        Args:
            path (pathlib.Path): The path where the class has been saved.

        Returns:
            `TFAlgo`: The class with the loaded elements.
        """
        checkpoint = self._update_from_checkpoint(path=path)
        assert (
            len(checkpoint) == 0
        ), f"Not all values from the checkpoint have been used: {checkpoint.keys()}"
        return self

    def _get_state_to_save(self) -> dict:
        """Create the algo checkpoint: a dictionary.
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
            "model_state_dict": self._model,
            "index_generator": self._index_generator,
        }
        if self._optimizer is not None:
            checkpoint["optimizer_state_dict"] = self._optimizer.get_config()
        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.get_config()

        return checkpoint

    def _check_tf_dataset(self):
        """Currently not used
        Check that the given Dataset is not an instance
        """
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

        # tf functions to save and load objects are quite different than torch's, we firstly try to copy the latter
        # with the pickle module
        with open(path, "wb") as f:
            cloudpickle.dump(self._get_state_to_save(), f)

        assert (
            path.is_file()
        ), f'Did not save the model properly {list(path.parent.glob("*"))}'

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
                    "type": "not visible yet",
                    # optimizer's parameters are not passed here because float32 are not json-serializable
                },
                "scheduler": None if self._scheduler is None else "not visible yet",
            }
        )
        return summary
