from typing import Generator
from typing import List

import tensorflow as tf

# we ignore batch normalization
# we ignore device handling


# model_parameters() -> model.weights attribute


# get_parameters -> get_weights() returns a copy


def increment_parameters(
    model: tf.keras.Model,
    updates: List[tf.Variable],
    updates_multiplier: float = 1.0,
) -> None:
    """Add the given update to the model parameters. This function modifies the given model internally and therefore returns nothing.

    Args:
        updates_multiplier (float, Optional): The coefficient which multiplies the updates before being added to the
            model. Defaults to 1.0.
    
    IMPORTANT THAT UPDATES HAS SAME DTYPE AS VARIABLE
    """
    n_parameters=len(model.weights)
    assert n_parameters == len(updates), "Length of model parameters and updates are unequal."

    for i in range(n_parameters):
        model.weights[i].assign_add(delta= updates_multiplier * updates[i])


def subtract_parameters(
    parameters: List[tf.Variable],
    parameters_to_subtract: List[tf.Variable],
) -> List[tf.Variable]:
    """
    subtract the given list of tf parameters i.e. : parameters - parameters_to_subtract.

    Args:
    Returns:
        typing.List[tf.Variable]: The subtraction of the given parameters.
    """
    return weighted_sum_parameters(
        parameters_list=[parameters, parameters_to_subtract],
        coefficient_list=[1, -1],
    )


def add_parameters(
    parameters: List[tf.Variable],
    parameters_to_add: List[tf.Variable],
) -> List[tf.Variable]:
    """
    add the given list of tf parameters i.e. : parameters + parameters_to_add.

    Args:
    Returns:
        typing.List[tf.Variable]: The addition of the given parameters.
    """
    return weighted_sum_parameters(
        parameters_list=[parameters, parameters_to_add],
        coefficient_list=[1, 1],
    )


def weighted_sum_parameters(
    parameters_list: List[List[tf.Variable]],
    coefficient_list: List[float],
) -> List[tf.Variable]:
    """
    Do a weighted sum of the given lists of tf parameters.
    Those elements can be extracted from a model thanks to the :func:`~get_weights` function.

    Args:
        parameters_list: [[org0_layer0_parameters, ...], [org1_layer0_parameters, ...], ...]
        coefficient_list (typing.List[float]): A list of coefficients which will be applied to each list of parameters.
    Returns:
        typing.List[tf.Variable]: The weighted sum of the given list of tf parameters.
    """

    weighted_sum = []

    assert all(
        len(parameters_list[0]) == len(parameters) for parameters in parameters_list
    ), "The number of parameters in each List is not the same"

    assert len(parameters_list) == len(coefficient_list), "There must be a coefficient for each List of parameters"

    for parameters_to_sum in zip(*parameters_list):
        assert all(
            parameters_to_sum[0].numpy().shape == parameter.numpy().shape for parameter in parameters_to_sum
        ), "The shape of the parameters are unequal."
        
        weighted_sum.append(sum(param * coeff for param, coeff in zip(parameters_to_sum, coefficient_list)))

    return weighted_sum


# set_parameters() -> set_weights() ?? yes


def zeros_like_parameters(
    model: tf.keras.Model,
) -> List[tf.Variable]:
    """Copy the model parameters from the provided tf model and sets values to zero.

    Args:

    Returns:
        typing.List[tf.Variable]: The list of torch parameters of the provided model
        with values set to zero.
    """
    parameters = []
    for layer_weights in model.weights:
        parameters.append(tf.zeros_like(input=layer_weights))
    
    return parameters
