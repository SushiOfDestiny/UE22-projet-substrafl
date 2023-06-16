import logging
from typing import Any
from typing import List
from typing import Optional

import tensorflow as tf

#from substrafl.algorithms.pytorch import weight_manager
#from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.index_generator import BaseIndexGenerator
from substrafl.remote import remote_data
from substrafl.schemas import FedAvgAveragedState
from substrafl.schemas import FedAvgSharedState
from substrafl.schemas import StrategyName


import tensorflow_algorithms.weight_manager as weight_manager
from tensorflow_algorithms.tf_base_algo import TFAlgo


logger = logging.getLogger(__name__)




class TFFedAvgAlgo(TFAlgo):
    """To be inherited. Wraps the necessary operation so a torch model can be trained in the Federated Averaging
    strategy.

    The ``train`` method:

        - updates the weights of the model with the aggregated weights,
        - initializes or loads the index generator,
        - calls the :py:func:`~substrafl.algorithms.tensorflow.TFFedAvgAlgo._local_train` method to do the local
          training
        - then gets the weight updates from the models and sends them to the aggregator.

    The ``predict`` method generates the predictions.

    The child class can override the :py:func:`~substrafl.algorithms.tensorflow.TFFedAvgAlgo._local_train` and
    :py:func:`~substrafl.algorithms.tensorflow.TFFedAvgAlgo._local_predict` methods, or other methods if necessary.

    To add a custom parameter to the ``__init__`` of the class, also add it to the call to ``super().__init__``
    as shown in the example with ``my_custom_extra_parameter``. Only primitive types (str, int, ...) are supported
    for extra parameters.

    Example: TODO
    """

    def __init__(
        self,
        model: tf.keras.Sequential,
        criterion: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        index_generator: BaseIndexGenerator,
        dataset: tf.data.Dataset,
        scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        # We ignore batch norm
        *args,
        **kwargs,
    ):
        """
        The ``__init__`` functions is called at each call of the `train()` or `predict()` function
        For round>=2, some attributes will then be overwritten by their previous states in the `load()` function,
        before the `train()` or `predict()` function is ran.

        Args:
            model (tf.keras.Sequential): A tensorflow model.
            criterion (tf.keras.losses.Loss): A tensorflow criterion (loss).
            optimizer (tf.keras.optimizers.Optimizer): A tensorflow optimizer linked to the model.
            index_generator (BaseIndexGenerator): a stateful index generator.
                Must inherit from BaseIndexGenerator. The __next__ method shall return a python object (batch_index)
                which is used for selecting each batch from the output of the get_data method of the opener
                during training in this way: ``x[batch_index], y[batch_index]``.
                If overridden, the generator class must be defined either as part of a package or in a different file
                than the one from which the ``execute_experiment`` function is called.
                This generator is used as stateful ``batch_sampler`` of the data loader created from the given
                ``dataset``
            dataset (tf.data.Dataset): an instantiable dataset class
            scheduler (tf.keras.optimizers.schedules.LearningRateSchedule, Optional): A tensorflow scheduler that will be called at every
                batch. If None, no scheduler will be used. Defaults to None.
            seed (typing.Optional[int]): Seed set at the algo initialization on each organization. Defaults to None.
            use_gpu (bool): Whether to use the GPUs if they are available. Defaults to True.
        """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=dataset,
            scheduler=scheduler,
            seed=seed,
            use_gpu=use_gpu,
            *args,
            **kwargs,
        )

    @property
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies

        Returns:
            typing.List: typing.List[StrategyName]
        """
        return [StrategyName.FEDERATED_AVERAGING]

    @remote_data
    def train(
        self,
        datasamples: Any,
        shared_state: Optional[FedAvgAveragedState] = None,  # Set to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
    ) -> FedAvgSharedState:
        """Train method of the fed avg strategy implemented with tensorflow. This method will execute the following
        operations:

            * instantiates the provided (or default) batch indexer
            * if a shared state is passed, set the parameters of the model to the provided shared state
            * train the model for n_updates
            * compute the weight update

        Args:
            datasamples (typing.Any): Input data returned by the ``get_data`` method from the opener.
            shared_state (FedAvgAveragedState, Optional): Dict containing tensorflow parameters that
                will be set to the model. Defaults to None.

        Returns:
            FedAvgSharedState: weight update (delta between fine-tuned
            weights and previous weights)
        """

        # Create tf dataset
        train_dataset = self._dataset(datasamples, is_inference=False)

        if shared_state is None:
            # Instantiate the index_generator
            assert self._index_generator.n_samples is None
            self._index_generator.n_samples = len(train_dataset)
        else:
            assert self._index_generator.n_samples is not None
            # The shared states is the average of the model parameter updates for all organizations
            # Hence we need to add it to the previous local state parameters
            with tf.device(self._device):
                # parameter_updates = [tf.convert_to_tensor(x) for x in shared_state.avg_parameters_update]
                parameter_updates = [tf.Variable(initial_value=x, dtype='float64') for x in shared_state.avg_parameters_update]
                weight_manager.increment_parameters(
                    model=self._model,
                    updates=parameter_updates,
                )

        self._index_generator.reset_counter()

        old_parameters = self._model.get_weights()

        # Train mode for tensorflow model
        self._model.train()

        # Train the model
        self._local_train(train_dataset)

        self._index_generator.check_num_updates()

        # Equivalent of self._model.eval() : desactivate the variables not used for prediction
        tf.keras.backend.set_learning_phase(0)

        parameters_update = weight_manager.subtract_parameters(
            parameters=self._model.get_weights(),
            parameters_to_subtract=old_parameters,
        )

        # Re set to the previous state
        self._model.set_weights(
            old_parameters,
        )

        with tf.device('CPU:0'):
            # https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
            parameters_updated = [tf.identity(p).numpy() for p in parameters_update]

            return FedAvgSharedState(
                n_samples=len(train_dataset),
                parameters_update=parameters_updated,
            )

    def summary(self):
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        # We ignore the summary's modification because it is only linked to the batch norm parameters
        return summary
