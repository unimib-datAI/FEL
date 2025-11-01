import os

import numpy as np
import tensorflow as tf
from fairlearn.metrics import demographic_parity_ratio
from ltn import Wrapper_Connective, Wrapper_Quantifier, fuzzy_ops
from ltn.core import Predicate, Variable
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Callable


def set_seed(seed: int):
    # Set seed for reproducibility
    SEED = seed

    # `PYTHONHASHSEED` environment variable
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # Python built-in random, numpy(+ scikit) and tensorflow seed
    tf.keras.utils.set_random_seed(SEED)

    # Enable TensorFlow op-determinism
    # from version 2.8 onwards https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)
    tf.config.experimental.enable_op_determinism()


def generate_attribute(Y, n_priv_pos, n_unpriv_pos, n_priv_neg, n_unpriv_neg):
    attr_pos = np.expand_dims(
        np.array([0] * n_priv_pos + [1] * n_unpriv_pos), axis=-1)
    np.random.shuffle(attr_pos)
    attr_neg = np.expand_dims(
        np.array([0] * n_priv_neg + [1] * n_unpriv_neg), axis=-1)
    np.random.shuffle(attr_neg)
    feature = np.copy(Y)
    feature[np.argwhere(Y == 1)] = attr_pos
    feature[np.argwhere(Y == 0)] = attr_neg
    return feature


def generate_sensitive_feature(Y, dp):
    """ Generate a sensitive feature

    Generate a sensitive feature with 2 categorical value (0, 1). The feature is generated
    to obtain dp: demographic_parity for the target variable Y

    """
    unpriv = np.sum(Y)
    priv = Y.shape[0] - unpriv
    unpriv_pos = unpriv - int(unpriv / 2)
    unpriv_neg = unpriv - unpriv_pos
    priv_pos = np.argwhere(Y == 1).shape[0] - unpriv_pos
    priv_neg = np.argwhere(Y == 0).shape[0] - unpriv_neg

    splits = np.array([priv_pos, unpriv_pos, priv_neg, unpriv_neg])

    dp = abs(1 - dp)
    feature = np.zeros(Y.shape)
    with tqdm() as pbar:
        while np.all(splits >= 0):
            feature = generate_attribute(
                Y, splits[0], splits[1], splits[2], splits[3])
            c_dp = abs(1 - demographic_parity_ratio(Y,
                    Y, sensitive_features=feature))
            if c_dp < dp:
                index = np.random.choice(4, 1, replace=False)
                splits[index] -= 1
                splits[index + (1 if index % 2 == 0 else -1)] += 1
                pbar.set_description(f"DP: {c_dp}", refresh=True)
                pbar.update()
            else:
                break

    return feature


def StandardScaleData(x_train, x_test):
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def StandardScaleData_ExcludingFeature(x_train, x_test, index):
    train_before = x_train[:, 0:index]
    train_col = x_train[:, index]
    train_after = x_train[:, index+1:]
    train = np.hstack([train_before, train_after])

    test_before = x_test[:, 0:index]
    test_col = x_test[:, index]
    test_after = x_test[:, index+1:]
    test = np.hstack([test_before, test_after])

    train, test, scaler = StandardScaleData(train, test)

    train_before = train[:, 0:index]
    train_after = train[:, index:]
    # col[:, None] mimick np.reshape()
    train = np.hstack([train_before, train_col[:, None], train_after])

    test_before = test[:, 0:index]
    test_after = test[:, index:]
    # col[:, None] mimick np.reshape()
    test = np.hstack([test_before, test_col[:, None], test_after])

    return train, test, scaler


class Oracle(object):
    # is a helper function to act as regular predicate predictor that outputs
    # true/false instead of fuzzy truth values.
    def __init__(self, oracle: Predicate, label_map: dict):
        self.oracle = oracle
        self.label_map = label_map

    def predict(self, data, logits=False):
        var_data = Variable("input", data)
        result = self.oracle(var_data)
        y_test_pred_prob = np.array(result.tensor)

        if logits:
            return y_test_pred_prob

        class_thresh = 0.5
        y_test_pred = np.zeros_like(y_test_pred_prob)
        y_test_pred[y_test_pred_prob >=
                    class_thresh] = self.label_map["positive"]
        y_test_pred[~(y_test_pred_prob >= class_thresh)
                    ] = self.label_map["negative"]
        return y_test_pred


class Eq():
    """Measure similarity"""

    def __call__(self, x, y):
        return tf.exp(-tf.norm(x-y))


class LTNOps(object):

    def __init__(self, implies_op: Callable, forall_deviation: int, aggregator_deviation: int) -> None:
        self._Not = Wrapper_Connective(fuzzy_ops.Not_Std())
        self._And = Wrapper_Connective(fuzzy_ops.And_Prod())
        self._Or = Wrapper_Connective(fuzzy_ops.Or_ProbSum())
        self._Implies = Wrapper_Connective(implies_op)
        self._Equiv = Wrapper_Connective(fuzzy_ops.Equiv(fuzzy_ops.And_Prod(), implies_op))
        self._deviation = forall_deviation
        self._Forall = Wrapper_Quantifier(fuzzy_ops.Aggreg_pMeanError(p=forall_deviation), semantics="forall")
        self._aggregator_deviation = aggregator_deviation
        self._formula_aggregator = fuzzy_ops.Aggreg_pMeanError(p=self._aggregator_deviation)

    @property
    def Not(self):
        return self._Not

    @property
    def And(self):
        return self._And

    @property
    def Or(self):
        return self._Or

    @property
    def Implies(self):
        return self._Implies

    @property
    def Equiv(self):
        return self._Equiv

    @property
    def ForAll(self):
        return self._Forall

    @property
    def FormulaAggregator(self):
        return self._formula_aggregator
    
    def get_description(self):
        return {
            "not": self._Not.connective_op.__class__.__name__,
            "and": self._And.connective_op.__class__.__name__,
            "or": self._Or.connective_op.__class__.__name__,
            "implies": self._Implies.connective_op.__class__.__name__,
            "equiv": self._Equiv.connective_op.__class__.__name__,
            "forall": {
                "aggregator": self._Forall.aggreg_op.__class__.__name__,
                "p": self._deviation
            },
            "formula_aggregator": {
                "aggregator": self._formula_aggregator.__class__.__name__,
                "p": self._aggregator_deviation
            }
        }
