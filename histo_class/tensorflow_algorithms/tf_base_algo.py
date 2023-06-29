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

        # self._device = self._get_tf_device(use_gpu=use_gpu) #does not exist

        # with tf.device(self._device):
            # self._optimizer = optimizer
            # # Move the optimizer to GPU if needed
            # #https://www.tensorflow.org/guide/gpu
            # if self._optimizer is not None:
            #     gpus = tf.config.list_physical_devices('GPU')
            #     if gpus:
            #         # Restrict TensorFlow to only use the first GPU
            #         try:
            #             tf.config.set_visible_devices(gpus, 'GPU')
            #             logical_gpus = tf.config.list_logical_devices('GPU')
            #         except RuntimeError as e:
            #             # Visible devices must be set before GPUs have been initialized
            #             print(e)

        # normally we receive here a dict of compiled model with the right config
        self._model = model

        self._optimizer = optimizer
        assert (self._optimizer is None), "Optimizer is not None"
        
        self._criterion = criterion
        
        self._scheduler = scheduler
        assert (self._scheduler is None), "Scheduler is not None"

        self._index_generator = index_generator
        assert (self._index_generator is None), "Index Generator is not None"

        self._dataset: tf.data.Dataset = dataset
        # dataset check overlooked

    @property
    def model(self) -> dict:
        """Model exposed when the user downloads the model

        Returns:
            dict: model
        """
        return self._model

    # def optimizer_deserialize(self) -> tf.keras.optimizers.Optimizer:
    #     """deserialize self._optimizer
    #     optimizer_class=tf.keras.optimizers.Adam(learning_rate=0.001)"""

    #     return tf.keras.optimizers.Adam(learning_rate=0.001).from_config(self._optimizer)

    # def optimizer_serialize(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
    #     """updates serialized optimizer self._optimizer with the state_dict of optimizer"""

    #     self._optimizer = optimizer.get_config()


    def model_deserialize(self) -> tf.keras.Sequential:
        """deserialize self._model
        first we try to put model in Sequential eventhough it is a custom CNN
        
        Returned model is compiled
        """

        # model = tf.keras.Sequential()
        # weight_manager.model_load_state_dict(model, self._model)

        # return model

        new_model = weight_manager.model_load_state_dict(self._model)
        
        # # Compiling
        # new_model.compile(optimizer=self.optimizer_deserialize(), loss=self._criterion)

        return new_model

    def model_serialize(self, model: tf.keras.Sequential) -> None:
        """updates serialized model self._model with the state_dict (config and weights) of model"""

        self._model = weight_manager.model_state_dict(model)

    
    # scheduler version in progress
    # def scheduler_deserialize(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    #     """deserialize self._scheduler
    #     scheduler_class=tf.keras.schedulers.Adam(learning_rate=0.001)"""

    #     return tf.keras.schedulers.Adam(learning_rate=0.001).from_config(self._scheduler)

    # def scheduler_serialize(self, scheduler: tf.keras.schedulers.scheduler) -> None:
    #     """updates serialized scheduler self._scheduler with the state_dict of scheduler"""

    #     self._scheduler = scheduler.get_config()

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
        self._local_predict(predict_dataset=predict_dataset,
                            predictions_path=predictions_path)

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
        # if self._index_generator is not None:
        #     # predict_loader = tf_dataloader(predict_dataset, batch_size=self._index_generator.batch_size)
        #     predict_loader = predict_dataset
        # else:
        #     raise BatchSizeNotFoundError(
        #         "No default batch size has been found to perform local prediction. "
        #         "Please overwrite the _local_predict function of your algorithm."
        #     )

        # predict_loader = predict_dataset

        # Equivalent of self._model.eval() : desactivate the variables not used for prediction
        tf.keras.backend.set_learning_phase(0)
        # Variable controlling the inference mode
        inference_mode = tf.Variable(True, trainable=False)

        # predictions = tf.constant([]) # 
        predictions = []

        # Deserialization and compiling
        model = self.model_deserialize()

        if inference_mode:
            # Code specific to the inference mode
            # with tf.device(self._device):
            # for x in predict_dataset.x[None, ...]: # taking np.array and adding a dimension
            #     # import ipdb
            #     # ipdb.set_trace()
            #     predictions = tf.concat([predictions, model(x)], 0)

            for i in range(len(predict_dataset)):
                x = predict_dataset[i]
                # y_logit = model(x)
                y = model(x)[0]# is a logit
                # y = tf.convert_to_tensor(value=np.argmax(y_logit))
                # predictions = tf.concat([predictions, y], 0)
                # predictions = tf.stack([predictions, y], 0)
                predictions.append(y)
            
            predictions = tf.convert_to_tensor(value=predictions)


        # with tf.device('CPU:0'):
        # https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
        predictions = tf.identity(predictions)
        self._save_predictions(predictions, predictions_path)

        # Reserialization, eventhough the model has not theoretically changed
        self.model_serialize(model)

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

        # # Train mode for tensorflow model

        # # Deserialization and compiling
        # model = self.model_deserialize()

        # model.train()

        # if self._optimizer is None:
        #     raise OptimizerValueError(
        #         "No optimizer found. Either give one or overwrite the _local_train method from the used torch"
        #         "algorithm."
        #     )
        # else:
        #     # Deserialization
        #     optimizer = self.optimizer_deserialize()

        # # Create tf dataloader
        # # train_data_loader = tf_dataloader(train_dataset, batch_sampler=self._index_generator) # reminder: class(train_dataset) = TFDataset / tf.data.Dataset
        # # avoiding the loader probleme
        # train_data_loader = train_dataset

        # # Changing device
        # # with tf.device(self._device):
        # for x_batch, y_batch in train_data_loader:

        #     # x_batch = x_batch.to(self._device)
        #     # y_batch = y_batch.to(self._device)

        #     # cf https://www.tensorflow.org/overview?hl=fr "For experts"
        #     # Forward pass
        #     y_pred = model(x_batch, training=True)

        #     # Compute loss
        #     loss = self._criterion(y_batch, y_pred)

        #     # Calculate gradients
        #     grads = tf.GradientTape.gradient(loss, model.trainable_variables)

        #     # Apply gradients
        #     optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #     if self._scheduler is not None:
        #         self._scheduler.step()

        # # Equivalent of self._model.eval() : desactivate the variables not used for prediction
        # tf.keras.backend.set_learning_phase(0)

        ###########################
        # Simplified training phase

        # Deserialization and compiling w/ optimizer and loss
        model = self.model_deserialize()

        # Create tf dataloader
        # train_data_loader = tf_dataloader(train_dataset, batch_sampler=self._index_generator) # reminder: class(train_dataset) = TFDataset / tf.data.Dataset
        # avoiding the loader probleme
        train_data_loader = train_dataset

        # We assume with the previous version, we can do the following
        model.fit(x=train_data_loader.x, y=train_data_loader.y, batch_size=32, epochs=1)

        
        # Reserialization
        # self._optimizer = model.get_compile_config()['optimizer']
        self.model_serialize(model)
        

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
        assert path.is_file(
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
            TorchAlgo: The class with the loaded elements.
        """
        checkpoint = self._update_from_checkpoint(path=path)
        assert len(
            checkpoint) == 0, f"Not all values from the checkpoint have been used: {checkpoint.keys()}"
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
            "model_state_dict": self._model,
            "index_generator": self._index_generator,
        }
        if self._optimizer is not None:
            # for an tf.optimizers.Optimizer, we use .get_config() and .from_config()
            checkpoint["optimizer_state_dict"] = self._optimizer
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

        # tf functions to save and load objects are quite different than torch's, we firstly try to imitate the latter
        # with the pickle module
        with open(path, "wb") as f:
            cloudpickle.dump(self._get_state_to_save(), f)

        assert path.is_file(
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
                    # "parameters": self._optimizer,
                },
                "scheduler": None if self._scheduler is None else "not visible yet",
            }
        )
        return summary
