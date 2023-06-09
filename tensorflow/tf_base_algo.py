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


        # device handling totally ignored atm
        self._optimizer = optimizer
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