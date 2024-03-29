from typing import Generator
from typing import List

import numpy as np
import tensorflow as tf

from collections import OrderedDict

# Hypothesis
# we ignore batch normalization
# we ignore device handling


def model_state_dict(model) -> OrderedDict:
    """Create a dict from a compiled model
    - the configuration (layers' structure)
    - the compile configuration (optimizer, loss, metric)
    - the parameters (weights and bias) of the model
    """
    s_dict = OrderedDict({})
    s_dict["config"] = model.get_config()
    s_dict["compile_config"] = model.get_compile_config()
    s_dict["weights"] = model.get_weights()

    return s_dict


def model_load_state_dict(s_dict) -> tf.keras.Sequential:
    """load the state dict into model
    Compile the model with compile_config infos
    """
    new_model = tf.keras.Sequential().from_config(s_dict["config"])
    new_model.compile_from_config(s_dict["compile_config"])
    new_model.set_weights(s_dict["weights"])

    return new_model


def increment_parameters(
    model: tf.keras.Model,
    updates: np.array,  # test with np.array and not np.array
    updates_multiplier: float = 1.0,
) -> None:
    """Add the given update to the model parameters. This function modifies the given model internally and therefore returns nothing.

    Args:
        updates_multiplier (float, Optional): The coefficient which multiplies the updates before being added to the
            model. Defaults to 1.0.

    Important:
        updates should have dtype='float32' because it is the float format chosen for the project
    """
    n_parameters = len(model.weights)
    assert n_parameters == len(
        updates
    ), "Length of model parameters and updates are unequal."

    # for i in range(n_parameters):
    #     model.weights[i].assign_add(delta=updates_multiplier * updates[i])

    current_parameters = model.get_weights()

    for i in range(n_parameters):
        current_parameters[i] = current_parameters[i] + updates_multiplier * updates[i]

    model.set_weights(current_parameters)


def subtract_parameters(
    parameters: List[np.array],
    parameters_to_subtract: List[np.array],
) -> List[np.array]:
    """
    subtract the given list of tf parameters i.e. : parameters - parameters_to_subtract.

    Args:
    Returns:
        typing.List[np.array]: The subtraction of the given parameters.
    """
    return weighted_sum_parameters(
        parameters_list=[parameters, parameters_to_subtract],
        coefficient_list=[1, -1],
    )


def add_parameters(
    parameters: List[np.array],
    parameters_to_add: List[np.array],
) -> List[np.array]:
    """
    add the given list of tf parameters i.e. : parameters + parameters_to_add.

    Args:
    Returns:
        typing.List[np.array]: The addition of the given parameters.
    """
    return weighted_sum_parameters(
        parameters_list=[parameters, parameters_to_add],
        coefficient_list=[1, 1],
    )


def weighted_sum_parameters(
    parameters_list: List[List[np.array]],
    coefficient_list: List[float],
) -> List[np.array]:
    """
    Do a weighted sum of the given lists of tf parameters.
    Those elements can be extracted from a model thanks to the :func:`~get_weights` function.

    Args:
        parameters_list: [[org0_layer0_parameters, ...], [org1_layer0_parameters, ...], ...]
        coefficient_list (typing.List[float]): A list of coefficients which will be applied to each list of parameters.
    Returns:
        typing.List[np.array]: The weighted sum of the given list of tf parameters.
    """

    weighted_sum = []

    assert all(
        len(parameters_list[0]) == len(parameters) for parameters in parameters_list
    ), "The number of parameters in each List is not the same"

    assert len(parameters_list) == len(
        coefficient_list
    ), "There must be a coefficient for each List of parameters"

    for parameters_to_sum in zip(*parameters_list):
        assert all(
            parameters_to_sum[0].shape == parameter.shape
            for parameter in parameters_to_sum
        ), "The shape of the parameters are unequal."

        weighted_sum.append(
            sum(
                param * coeff
                for param, coeff in zip(parameters_to_sum, coefficient_list)
            )
        )

    return weighted_sum


def zeros_like_parameters(
    model: tf.keras.Model,
) -> List[np.array]:
    """Copy the model parameters from the provided tf model and sets values to zero.

    Args:

    Returns:
        typing.List[np.array]: The list of tf parameters of the provided model
        with values set to zero.
    """
    parameters = model.get_weights()
    for layer in parameters:
        layer = 0.0

    return parameters
