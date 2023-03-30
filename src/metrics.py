from torch import Tensor
import torch


def l0_norm(w: Tensor) -> Tensor:
    """
    Calculate l0 norm of a vector

    :param w: Weights of regression
    :return: Amount of non-zero elements in a weights vector
    :rtype: Tensor
    """
    return torch.count_nonzero(w)


def l1_norm(w: Tensor) -> Tensor:
    """
    Calculate l1 norm of a vector

    :param w: Weights of regression
    :return: l1 norm for given weights
    :rtype: Tensor
    """
    return torch.norm(w, p=1)


def l2_norm(w: Tensor) -> Tensor:
    """
    Calculate l2 norm of a vector

    :param w: Weights of regression
    :return: l2 norm for given weights
    :rtype: Tensor
    """
    return torch.norm(w, p=2)


def mean_squared_error(y_hat: Tensor, y_true: Tensor, w: Tensor = None, lambda_: float = 0) -> Tensor:
    """
    Calculate MSE on the given data

    :param Tensor y_hat: Predicted values for the target variables
    :param Tensor y_true: True values for the target variables
    :param Tensor w: Made for compatibility with lasso and ridge losses
    :param float lambda_: Made for compatibility with lasso and ridge losses
    :return: MSE
    :rtype: Tensor
    """
    return torch.mean((y_hat - y_true) ** 2)


def lasso(y_hat: Tensor, y_true: Tensor, w: Tensor, lambda_: float) -> Tensor:
    """
    Calculate lasso optimization goal value

    :param Tensor y_hat: Predicted values for the target variables
    :param Tensor y_true: True values for the target variables
    :param Tensor w: Weights of given regression model
    :param float lambda_: Regularization coefficient
    :return: Lasso optimization goal w.r.t. current weights
    :rtype: Tensor
    """
    return mean_squared_error(y_hat, y_true) + lambda_ * l1_norm(w)


def ridge(y_hat: Tensor, y_true: Tensor, w: Tensor, lambda_: float) -> Tensor:
    """
    Calculate ridge optimization goal value

    :param Tensor y_hat: Predicted values for the target variables
    :param Tensor y_true: True values for the target variables
    :param Tensor w: Weights of given regression model
    :param float lambda_: Regularization coefficient
    :return: Ridge optimization goal w.r.t. current weights
    :rtype: Tensor
    """
    return mean_squared_error(y_hat, y_true) + lambda_ / 2 * l2_norm(w) ** 2
