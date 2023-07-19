import logging
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf

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
        criterion: tf.keras.losses.Loss, # set to None
        optimizer: tf.keras.optimizers.Optimizer, # set to None
        index_generator: Union[
            BaseIndexGenerator, None
        ],  # set to None
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
            model (dict): the state_dict, or global configuration of a tensorflow model (config, compile_config, weights).
            criterion (tf.keras.losses.Loss): A tensorflow criterion (loss).
            optimizer (tf.keras.optimizers.Optimizer): A tensorflow optimizer linked to the model.
            index_generator (BaseIndexGenerator): a stateful index generator.
            dataset (tf.data.Dataset): an instantiable dataset class
            scheduler (tf.keras.optimizers.schedules.LearningRateSchedule, Optional): A tensorflow scheduler that will be called at every
                batch. If None, no scheduler will be used. Defaults to None.
            seed (typing.Optional[int]): Seed set at the algo initialization on each organization. Defaults to None.
            use_gpu (bool): Whether to use the GPUs if they are available. Defaults to True.
        """

        # initialized exactly like TFAlgo
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
        shared_state: Optional[
            FedAvgAveragedState
        ] = None,  # Set to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
    ) -> FedAvgSharedState:
        """Train method of the fed avg strategy implemented with tensorflow. This method will execute the following
        operations:

            * if a shared state is passed, set the parameters of the model to the provided shared state
            * train the model for n_updates
            * compute the weight update

        Args:
            datasamples (typing.Any): Input data returned by the ``x`` and ``y`` methods from the opener.
            shared_state (FedAvgAveragedState, Optional): Dict containing tensorflow parameters that
                will be set to the model. Defaults to None.

        Returns:
            FedAvgSharedState: weight update (delta between fine-tuned
            weights and previous weights)
        """

        # Create TFDataset instance
        train_dataset = self._dataset(datasamples, is_inference=False)

        if shared_state is None:
            if self._index_generator is not None: # normally set to None
                # Instantiate the index_generator
                assert self._index_generator.n_samples is None
                self._index_generator.n_samples = len(train_dataset)
        else:
            if self._index_generator is not None:
                assert self._index_generator.n_samples is not None
            # The shared states is the average of the model parameter updates for all organizations
            # Hence we need to add it to the previous local state parameters
            
            # parameter_updates = [
            #     tf.Variable(initial_value=x, dtype="float32")
            #     for x in shared_state.avg_parameters_update
            # ]

            parameters_updates = shared_state.avg_parameters_update

            # A simpler version of following code is possible because increment_parameter
            # needs the object model, but its weights are enough

            # Deserializing and compiling
            model = self.model_deserialize()

            weight_manager.increment_parameters(
                model=model,
                updates=parameters_updates,
            )

            # Reserializing
            self.model_serialize(model)

        old_parameters = self._model["weights"]

        # Train the model
        self._local_train(train_dataset)

        if self._index_generator is not None:
            self._index_generator.check_num_updates()

        parameters_update = weight_manager.subtract_parameters(
            parameters=self._model["weights"],
            parameters_to_subtract=old_parameters,
        )

        print(f'Variation des poids selon le weight_manager : {parameters_update[2][0][0][0][:5]}')

        # Re set to the previous state
        self._model["weights"] = old_parameters

        return FedAvgSharedState(
            n_samples=len(train_dataset),
            parameters_update=parameters_update, # maybe we will have to convert parameters_update in List(tf.Variable)
        )

    def summary(self):
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        
        return summary
