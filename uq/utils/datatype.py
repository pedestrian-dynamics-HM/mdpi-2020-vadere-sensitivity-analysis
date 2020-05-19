import numpy as np
import warnings


def is_vector(v) -> bool:
    if type(v) is not np.ndarray:
        v = np.ndarray(v)
        warnings.warn('Input to is_vector should be of type np.ndarray')
    if np.sum(np.asarray(v.shape) > 1) == 1:  # more than one element in only one direction
        bool_return = True
    else:
        bool_return = False

    return bool_return


def is_matrix(v) -> bool:
    if np.sum(np.asarray(v.shape) > 1) == 2:  # more than one element in 2 directions
        bool_return = True
    else:
        bool_return = False

    return bool_return


def is_row_vector(v) -> bool:
    if is_vector(v) is True:
        if type(v) == list:
            v = np.array(v)
        if type(v) == np.ndarray and np.ndim(v) == 2 and np.size(v, axis=1) > 1:
            bool_return = True
        elif type(v) == np.ndarray and np.ndim(v) == 1 and np.size(v, axis=0) > 1:
            bool_return = True
        else:
            bool_return = False
    else:
        bool_return = False
    return bool_return


def test_is_vector():
    # scalars
    assert is_vector(1) is False
    assert is_vector(np.array(1)) is False
    assert is_vector(np.array([1])) is False
    assert is_vector(np.array([[1]])) is False
    assert is_vector([1]) is False

    # vectors
    assert is_vector(np.array([1, 2])) is True
    assert is_vector(np.array([1, 2, 3, 4, 5])) is True
    assert is_vector(np.array([[1, 2]])) is True
    assert is_vector([1, 2]) is True
    assert is_vector(np.array([[1], [2]])) is True

    # matrices
    assert is_vector(np.array([[1, 2, 3], [4, 5, 6]])) is False


def test_is_row_vector():
    # vectors
    assert is_row_vector(np.array([1, 2])) is True
    assert is_row_vector(np.array([1, 2, 3, 4, 5])) is True
    assert is_row_vector(np.array([[1, 2]])) is True
    assert is_row_vector([1, 2]) is True
    assert is_row_vector([[1, 2]]) is True

    # column vectors
    assert is_row_vector(np.array([[1], [2]])) is False
    assert is_row_vector([[1], [2]]) is False


def test_is_scalar():
    assert is_scalar(1) is True
    assert is_scalar(np.array(1)) is True
    assert is_scalar(np.array([1])) is True
    assert is_scalar(np.array([[1]])) is True
    assert is_scalar([1]) is True


def is_scalar(v) -> bool:
    if np.size(v) == 1:
        bool_return = True
    else:
        bool_return = False

    return bool_return


"""" ----------------------------------- boxing / unboxing  --------------------------------------------------- """


def unbox(parameter_value):
    if is_scalar(parameter_value):
        while type(parameter_value) == np.ndarray:
            parameter_value = parameter_value.item()

    if type(parameter_value) == np.ndarray:
        if any(np.array(np.shape(parameter_value)) == 1):
            parameter_value = parameter_value.flatten()

    if type(parameter_value) == list:
        if len(parameter_value) == 1:
            # todo: check if this works in all cases [[1,2]] and [[1],[2]]
            parameter_value = parameter_value[0]

    return parameter_value


def box(parameter_value):
    if type(parameter_value) == int or type(parameter_value) == float or type(parameter_value) == np.float64:
        parameter_value = np.ones(shape=(1, 1)) * parameter_value
    elif type(parameter_value) == np.ndarray and np.ndim(parameter_value) < 2:
        parameter_value = np.expand_dims(parameter_value, axis=1)

    if type(parameter_value) == bool:
        parameter_value = [parameter_value]

    return parameter_value


def box_if_scalar(parameter_value):
    if np.ndim(parameter_value) == 0:
        parameter_value = np.array(parameter_value)
    return parameter_value


def unbox_if_scalar(parameter_value):
    if is_scalar(parameter_value) and np.ndim(parameter_value) > 0:
        parameter_value = parameter_value[0]
    return parameter_value


def box_to_n_dim(parameter_value, dim: int):
    if np.ndim(parameter_value) < dim:
        i = np.ndim(parameter_value)
        while i < dim:
            parameter_value = np.array([parameter_value])
            i = i + 1
    elif np.ndim(parameter_value) > dim:
        raise Warning("Reducing of dimensions not implemented yet")

    return parameter_value


def box1d(parameter_value):
    if is_scalar(parameter_value) and np.ndim(parameter_value) == 0:
        parameter_value = np.array([parameter_value])

    return parameter_value


def get_dimension(key):
    if type(key) == list:
        dim = len(key)
    elif type(key) == str:
        dim = 1
    else:
        raise Exception('Type of key must be either a string or a list')
    return dim


def nr_entries(input_vector):
    if np.ndim(input_vector) == 0:
        number = 1
    else:
        number = input_vector.size
    return number


""" ----------------------------------- dealing with vectors --------------------------------------------------"""


def assure_vec(surrogate_eval):
    """ assure that input is a vector, if it is a scalar, convert to vector, if it is a matrix, print warning """

    if np.ndim(surrogate_eval) == 0:
        surrogate_eval = np.array([surrogate_eval])

    if not is_vector(surrogate_eval) and np.ndim(surrogate_eval) > 1:
        raise Warning('assure_vec: input Matrix cannot be converted to vector')

    return surrogate_eval


def length(input_vector):
    if is_scalar(input):
        length_input = 1
    else:
        length_input = len(input_vector)

    return length_input
