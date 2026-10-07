"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers
import random

import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        # return np.abs(corr)
        return max(0, corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)

def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))

def _temp(y, y_pred, w):
    """Calculate the root mean square error."""

    # y_range_q1 = min(y) + (np.ptp(y) * 0.25)
    # y_range_q3 = min(y) + (np.ptp(y) * 0.75)
    # pred_range_q1 = min(y_pred) + (np.ptp(y_pred) * 0.25)
    # pred_range_q3 = min(y_pred) + (np.ptp(y_pred) * 0.75)
    # mask_range_q1 = (np.sign(y - y_range_q1) == np.sign(y_pred - pred_range_q1)).astype(int)
    # mask_range_q3 = (np.sign(y - y_range_q3) == np.sign(y_pred - pred_range_q3)).astype(int)
    # da_range_q1, da_range_q3 = np.average(mask_range_q1, weights=w), np.average(mask_range_q3, weights=w)

    # y_q1, y_q2, y_q3                = _weighted_quantile(y, w, 0.25), _weighted_quantile(y, w, 0.50), _weighted_quantile(y, w, 0.75)
    # y_pred_q1, y_pred_q2, y_pred_q3 = _weighted_quantile(y_pred, w, 0.25), _weighted_quantile(y_pred, w, 0.50), _weighted_quantile(y_pred, w, 0.75)
    # mask_q1 = (np.sign(y - y_q1) == np.sign(y_pred - y_pred_q1)).astype(int)
    # mask_q2 = (np.sign(y - y_q2) == np.sign(y_pred - y_pred_q2)).astype(int)
    # mask_q3 = (np.sign(y - y_q3) == np.sign(y_pred - y_pred_q3)).astype(int)
    # da_q1, da_q2, da_q3 = np.average(mask_q1, weights=w), np.average(mask_q2, weights=w), np.average(mask_q3, weights=w)

    # y_mean = np.average(y, weights=w)
    # y_pred_mean = np.average(y_pred, weights=w)
    # mask_mean = (np.sign(y - y_mean) == np.sign(y_pred - y_pred_mean)).astype(int)
    # da_mean = 1 - np.average(mask_mean, weights=w)
    #
    # y_range = max(y) - min(y)
    # y_pred_range = max(y_pred) - min(y_pred)
    # range_diff = max(0, abs(y_range - y_pred_range)/ y_range)

    # y_q2        =  _weighted_quantile(y, w, 0.50)
    # y_pred_q2   = _weighted_quantile(y_pred, w, 0.50)
    # mask_q2     = (np.sign(y - y_q2) == np.sign(y_pred - y_pred_q2)).astype(int)
    # da_q2       = np.average(mask_q2, weights=w)

    # directional_accuracy = (min(da_mean, da_range_q1, da_range_q3, da_q1, da_q2, da_q3) * min(max(0, range_diff), max(0, range_inti)))
    # mean_acc    = min(da_range_q1, da_mean ,da_range_q3)
    # quater_acc  = min(da_q1, da_q2, da_q3)
    # range_acc   = np.sqrt(max(0, range_diff) * max(0, range_inti))

    # directional_accuracy = max(0,
    #                            (3 * da_mean * min(da_q1, da_q3) * range_acc) / (da_mean + min(da_q1, da_q3) + range_acc + 1e-10)
    #                            )
    # directional_accuracy = max(0, da_mean + min(da_q1, da_q2) + range_inti + range_diff) * (1/4)

    # y_mean_weighted = np.average(y, weights=w)
    # numerator = np.sqrt(np.average((y - y_pred) ** 2, weights=w))
    # denominator = np.sqrt(np.average((y - y_mean_weighted) ** 2, weights=w))
    # r2 = 10 - (numerator / denominator) if denominator != 0 else 0.0
    # r2 = (2 ** max(0, r2)) / 1024
    # r2_max = 1 - (numerator / denominator) if denominator != 0 else 0.0
    # r2_max = max(0, r2_max)

    # return_value = (2 * r2_score * directional_accuracy) / (r2_score + directional_accuracy + 1e-10)
    # return_value = 0.75 * r2_score + 0.3 * directional_accuracy

    # return_value = 0.70 * max(0, r2_score) + 0.30 * da_mean
    # return_value = (2 * max(0, r2_score) * directional_accuracy) / (max(0, r2_score) + directional_accuracy + 1e-10)

    # return 0.7 * r2 + 0.3 * np.sqrt(da_mean * range_diff)
    rmse = _root_mean_square_error(y, y_pred, w)
    spearman = max(0.30, (1 - _weighted_spearman(y, y_pred, w)))
    return rmse * (0.70 + spearman)

    # return ((2 * r2 * directional_accuracy) / (r2 + directional_accuracy + 1e-10)) + r2_max

    # return np.sqrt((pearson_value + diff_value) * (rmse_value + diff_value))
    # sample        = (3/4) * max(pearson_value, rmse_value) + (1/4) * min(pearson_value, rmse_value)
    # sample        = average_value + 0.25 * diff_value
    # min_diff      = (1 + pearson_value) * (1 + abs(_r2_score(y, y_pred, w) - 1)) * denominator

def _weighted_quantile(data, weights, quantile):
    """Computes a weighted quantile (e.g., 0.25 for Q1, 0.75 for Q3)."""
    sorter = np.argsort(data)
    sorted_data = data[sorter]
    sorted_weights = weights[sorter]

    cum_weights = np.cumsum(sorted_weights)
    cutoff = quantile * cum_weights[-1]
    return sorted_data[np.searchsorted(cum_weights, cutoff)]

def _r2_score(y, y_pred, w):
    """Calculate the weighted R2 (Coefficient of Determination) score."""
    y_mean_weighted = np.average(y, weights=w)
    numerator = np.average((y - y_pred) ** 2, weights=w)
    denominator = np.average((y - y_mean_weighted) ** 2, weights=w)
    # Safeguard against zero variance (division by zero)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)


weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)
r2_score = _Fitness(function=_r2_score,
                            greater_is_better=False)
temp = _Fitness(function=_temp,
                            greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss,
                'r2 score': r2_score,
                'temp': temp}
