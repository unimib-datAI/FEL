import json
import itertools
from typing import List, Optional, Type, Tuple

import ltn
import tensorflow as tf
from ltn.core import Expression
from sklearn.metrics import classification_report, confusion_matrix
from numpy import vstack
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference, equalized_odds_ratio, equalized_odds_difference
import numpy as np

from AxiomDataClass import Axiom, AxiomConfig, AxiomBase, Variable
from utils import LTNOps, Oracle

class KnowledgeBase(object):

    def __init__(
            self,
            Xtrain,
            Xtest,
            Ytrain,
            Ytest,
            label_map,
            privileged_value,
            unprivileged_value,
            hidden_layer_sizes,
            fuzzy_ops: LTNOps,
            sensitive_feature_index = None,           
            config_file='./KnowledgeBaseAxioms.json') -> None:

        self._Xtrain = Xtrain
        self._Xtest = Xtest
        self._Ytrain = Ytrain
        self._Ytest = Ytest
        self._label_map = label_map
        self._pv = privileged_value
        self._uv = unprivileged_value
        self._hidden_layer_sizes = hidden_layer_sizes
        self._fuzzy_ops = fuzzy_ops

        self._sfi = sensitive_feature_index
        if self._sfi == None:
            self._sfi = self._Xtrain.shape[1] - 1

        self._Xtrain_priv = Xtrain[Xtrain[:, self._sfi] == self._pv]
        self._Xtrain_upriv = Xtrain[Xtrain[:, self._sfi] == self._uv]
        self._Ytrain_priv = Ytrain[Xtrain[:, self._sfi] == self._pv]
        self._Ytrain_upriv = Ytrain[Xtrain[:, self._sfi] == self._uv]

        self._Xtest_priv = Xtest[Xtest[:, self._sfi] == self._pv]
        self._Xtest_upriv = Xtest[Xtest[:, self._sfi] == self._uv]
        self._Ytest_priv = Ytest[Xtest[:, self._sfi] == self._pv]
        self._Ytest_upriv = Ytest[Xtest[:, self._sfi] == self._uv]

        # Connectives and quantifiers
        self.__init_conn_quant__()

        # Variables
        self.__init_variables__()

        # Predicates
        self.__init_predicates__()

        # Axioms
        self.__init_axioms__()

        # Knowledge base config
        self._config = self._get_configured_axioms(config_file)

        # Predictor
        self._oracle = Oracle(self._predicate_classifier, label_map)

    def __init_conn_quant__(self):
        self._Not = self._fuzzy_ops.Not
        self._And = self._fuzzy_ops.And
        self._Or = self._fuzzy_ops.Or
        self._Implies = self._fuzzy_ops.Implies
        self._Equiv = self._fuzzy_ops.Equiv
        self._Forall = self._fuzzy_ops.ForAll
        self._formula_aggregator = self._fuzzy_ops.FormulaAggregator

        class EquivSim():
            """Measure similarity"""

            def __call__(self, x, y):
                return tf.exp(-tf.norm(x-y))

        self._EquivSim = ltn.Wrapper_Connective(EquivSim())

    def __init_variables__(self):
        self._var_all = ltn.Variable(
            "all", vstack([self._Xtrain, self._Xtest]))
        self._var_train_all = ltn.Variable("train_all", self._Xtrain)
        self._var_test_all = ltn.Variable("test_all", self._Xtest)
        self._var_train_positive = ltn.Variable(
            "train_positive", self._Xtrain[self._Ytrain == self._label_map['positive']])
        self._var_train_negative = ltn.Variable(
            "train_negative", self._Xtrain[self._Ytrain == self._label_map['negative']])
        self._var_test_positive = ltn.Variable(
            "test_positive", self._Xtest[self._Ytest == self._label_map['positive']])
        self._var_test_negative = ltn.Variable(
            "test_negative", self._Xtest[self._Ytest == self._label_map['negative']])

    def __init_predicates__(self):
        self._predicate_classifier = ltn.Predicate.MLP(  # type: ignore
            [self._Xtrain.shape[1]], hidden_layer_sizes=self._hidden_layer_sizes)
        self._predicate_privileged = ltn.Predicate.Lambda(  # type: ignore
            lambda x: x[:, self._sfi] == self._pv)
        self._predicate_unprivileged = ltn.Predicate.Lambda(    # type: ignore
            lambda x: x[:, self._sfi] == self._uv)
        self._predicate_positive = ltn.Predicate.Lambda( # type: ignore
            lambda x: self._Ytrain if (x.shape[0] == self._Xtrain.shape[0]) else self._Ytest)
        self._predicate_negative = ltn.Predicate.Lambda( # type: ignore
            lambda x: tf.logical_not(self._Ytrain) if (x.shape[0] == self._Xtrain.shape[0]) else tf.logical_not(self._Ytest))
        self._pradicate_oracle = ltn.Predicate.Lambda( # type: ignore
            lambda x: tf.cast(tf.math.greater_equal(self._predicate_classifier(ltn.Variable('x', x)).tensor, 0.5), tf.float32) 
        )

    def __init_axioms__(self):
        # Classification axioms
        self._axiom_positive_class = lambda x: self._Forall(
            x, self._predicate_classifier(x))
        self._axiom_negative_class = lambda x: self._Forall(
            x, self._Not(self._predicate_classifier(x)))

        self._axiom_positive_class_all = lambda x: self._Forall(
            x, self._Implies(
                self._predicate_positive(x),
                self._predicate_classifier(x)
            ) 
        )
        self._axiom_negative_class_all = lambda x: self._Forall(
            x, self._Implies(
                self._predicate_negative(x),
                self._Not(self._predicate_classifier(x)) 
            ) 
        )

        # Demographic parity axioms defined on all samples
        self._axiom_privileged = lambda x: self._Forall(x, self._Implies(
            self._predicate_privileged(x), self._predicate_classifier(x)))
        self._axiom_unprivileged = lambda x: self._Forall(x, self._Implies(
            self._predicate_unprivileged(x), self._predicate_classifier(x)))
        self._axiom_demographic_parity = lambda x: self._Equiv(
            self._axiom_privileged(x), self._axiom_unprivileged(x))
        
        # Demographic parity axioms where the classifier predicate outputs 1s or 0s
        self._axiom_demographic_parity_oracle = lambda x: self._Equiv(
            self._Forall(x, self._Implies(self._predicate_privileged(x), self._pradicate_oracle(x))),
            self._Forall(x, self._Implies(self._predicate_unprivileged(x), self._pradicate_oracle(x)))
        )

        # Demographic parity axioms splitted between positve and negative samples
        self._axiom_privileged_pos = lambda x: self._Forall(x, self._Implies(
            self._predicate_privileged(x), self._predicate_classifier(x)))
        self._axiom_unprivileged_pos = lambda x: self._Forall(x, self._Implies(
            self._predicate_unprivileged(x), self._predicate_classifier(x)))
        self._axiom_privileged_neg = lambda x: self._Forall(x, self._Implies(
            self._predicate_privileged(x), self._Not(self._predicate_classifier(x))))
        self._axiom_unprivileged_neg = lambda x: self._Forall(x, self._Implies(
            self._predicate_unprivileged(x), self._Not(self._predicate_classifier(x))))
        self._axiom_demographic_parity_pos = lambda x: self._Equiv(
            self._axiom_privileged_pos(x), self._axiom_unprivileged_pos(x))
        self._axiom_demographic_parity_neg = lambda x: self._Equiv(
            self._axiom_privileged_neg(x), self._axiom_unprivileged_neg(x))

        # Demographic parity axioms with guarded quantifiers and similarity equivalence
        self._axiom_privileged_guarded = lambda x: self._Forall(
            x, self._predicate_classifier(x), mask=self._predicate_privileged(x))
        self._axiom_unprivileged_guarded = lambda x: self._Forall(
            x, self._predicate_classifier(x), mask=self._predicate_unprivileged(x))
        self._axiom_demographic_parity_sim = lambda x: self._EquivSim(
            self._axiom_privileged_guarded(x), self._axiom_unprivileged_guarded(x))

    def _get_configured_axioms(self, config_file: str) -> List[Axiom]:
        with open(config_file) as json_file:
            axiomsConfigs: List[AxiomConfig] = AxiomConfig.schema().loads(
                json_file.read(), many=True)
            axioms = [
                Axiom(
                    config.name, config.infos,
                    getattr(self, f'_{config.name}'),
                    self._get_axiom_variables(config)
                )
                for config
                in axiomsConfigs]
            return axioms

    def _get_training_axioms(self) -> List[Axiom]:
        return [
            axiom
            for axiom
            in self._config
            if axiom.infos.training
        ]

    def _aggregate(self, axioms_sats: List[Tuple[Optional[Expression], float]]):
        weights = [weight for ax, weight in axioms_sats if ax is not None]
        if weights:
            axioms = tf.stack(
                [
                    tf.squeeze(ax.tensor)
                    for ax, _
                    in axioms_sats
                    if ax is not None
                ]
            )
            weighted_axioms = weights*axioms  # type: ignore
            return self._formula_aggregator(weighted_axioms)
        return None

    def _get_axiom_variables(self, axiom: AxiomConfig) -> List[Optional[Variable]]:
        variables = (
            Variable(
                name=f'var_{split_name}_{var_name}',
                training=split_name == 'train',
                var=getattr(self, f'_var_{split_name}_{var_name}', None)
            )
            for var_name, split_name
            in itertools.product(axiom.infos.variables, axiom.infos.splits)
        )
        return [v for v in variables if v is not None]

    def _get_axiom_sat(self, axiom: Axiom, split: str) -> Optional[Expression]:
        vars = []
        for var_name in axiom.infos.variables:
            var = getattr(self, f'_var_{split}_{var_name}', None)
            if var is not None:
                vars.append(var)
        if vars:
            return axiom.ltnAxiom(*vars)
        return None

    @property
    def trainable_variables(self):
        return self._predicate_classifier.trainable_variables

    @property
    def config(self) -> dict:
        jsonConfigString = AxiomConfig.schema().dumps([AxiomConfig(
            ax.name, ax.infos) for ax in self._config], many=True)   # type: ignore
        return json.loads(jsonConfigString)

    @property
    def axioms(self) -> dict:
        return {
            axiom['name']: axiom['infos']
            for axiom
            in self.config
        }

    @property
    def weights(self) -> list:
        return [
            axiom.infos.weight
            for axiom
            in self._config
            if axiom.infos.training
        ]

    @tf.function
    def train_step(self, split: str = 'train') -> Optional[tf.Tensor]:
        # Get training axioms
        training_axioms = self._get_training_axioms()
        # Evaluate them for the specified split
        training_axioms_sats = [
            (self._get_axiom_sat(axiom, split), axiom.infos.weight)
            for axiom in training_axioms]
        # Calculate satisfability
        sat = self._aggregate(training_axioms_sats)
        return sat

    def get_logs(self) -> dict:
        
        reported_axioms = ['axiom_positive_class', 'axiom_negative_class', 'axiom_privileged', 'axiom_unprivileged', 'axiom_demographic_parity']
        axiom_sat_report = {}

        # All axioms satisfability
        # for axiom in self._config:
        # Only in reported (better performance in evaluation)
        for axiom in [a for a in self._config if a.name in reported_axioms]:
            axiom_sat_report[f'{axiom.name}_sat'] = {}
            if axiom.infos.weight > 0:
                axiom_sat_report[f'{axiom.name}_weighted_sat'] = {}
            for split in axiom.infos.splits:
                sat = self._get_axiom_sat(axiom, split)
                if sat:
                    axiom_sat_report[f'{axiom.name}_sat'][split] = sat.tensor.numpy()
                    if axiom.infos.weight > 0:
                        axiom_sat_report[f'{axiom.name}_weighted_sat'][split] = sat.tensor.numpy() * axiom.infos.weight

        # Trained axioms satisfability
        training_axioms = self._get_training_axioms()
        splits = set.intersection(*[set(axiom.infos.splits)
                                  for axiom in training_axioms])
        training_axioms_sat_report = {
            f'{split}_sat': sat.numpy()
            for split
            in splits
            if (sat := self.train_step(split)) is not None
        }

        # Classification metrics
        train_pred = self._oracle.predict(self._Xtrain)
        train_report = {
            'train_classification_metrics': classification_report(
                self._Ytrain, train_pred, output_dict=True
            )
        }
        test_pred = self._oracle.predict(self._Xtest)
        test_report = {
            'test_classification_metrics': classification_report(
                self._Ytest, test_pred, output_dict=True
            )
        }

        # Confusion matrix
        confusion_matrix_keys = ["tn", "fp", "fn", "tp"]
        confusion_matrix_report = {
            "train": {
                "all": dict(zip(
                    confusion_matrix_keys, 
                    confusion_matrix(self._Ytrain, train_pred).ravel())),
                "privileged": dict(zip(
                    confusion_matrix_keys, 
                    confusion_matrix(self._Ytrain_priv, self._oracle.predict(self._Xtrain_priv)).ravel())),
                "unprivileged": dict(zip(
                    confusion_matrix_keys, 
                    confusion_matrix(self._Ytrain_upriv, self._oracle.predict(self._Xtrain_upriv)).ravel()))
            },
            "test": {
                "all": dict(zip(
                    confusion_matrix_keys, 
                    confusion_matrix(self._Ytest, test_pred).ravel())),
                "privileged": dict(zip(
                    confusion_matrix_keys, 
                    confusion_matrix(self._Ytest_priv, self._oracle.predict(self._Xtest_priv)).ravel())),
                "unprivileged": dict(zip(
                    confusion_matrix_keys, 
                    confusion_matrix(self._Ytest_upriv, self._oracle.predict(self._Xtest_upriv)).ravel()))
            }
        }

        # Add predicted postitve
        confusion_matrix_report["train"]["all"]["pp"] = confusion_matrix_report["train"]["all"]["tp"] + confusion_matrix_report["train"]["all"]["fp"]
        confusion_matrix_report["train"]["privileged"]["pp"] = confusion_matrix_report["train"]["privileged"]["tp"] + confusion_matrix_report["train"]["privileged"]["fp"]
        confusion_matrix_report["train"]["unprivileged"]["pp"] = confusion_matrix_report["train"]["unprivileged"]["tp"] + confusion_matrix_report["train"]["unprivileged"]["fp"]
        confusion_matrix_report["test"]["all"]["pp"] = confusion_matrix_report["test"]["all"]["tp"] + confusion_matrix_report["test"]["all"]["fp"]
        confusion_matrix_report["test"]["privileged"]["pp"] = confusion_matrix_report["test"]["privileged"]["tp"] + confusion_matrix_report["test"]["privileged"]["fp"]
        confusion_matrix_report["test"]["unprivileged"]["pp"] = confusion_matrix_report["test"]["unprivileged"]["tp"] + confusion_matrix_report["test"]["unprivileged"]["fp"]

        # Add predicted negatives
        confusion_matrix_report["train"]["all"]["pn"] = confusion_matrix_report["train"]["all"]["tn"] + confusion_matrix_report["train"]["all"]["fn"]
        confusion_matrix_report["train"]["privileged"]["pn"] = confusion_matrix_report["train"]["privileged"]["tn"] + confusion_matrix_report["train"]["privileged"]["fn"]
        confusion_matrix_report["train"]["unprivileged"]["pn"] = confusion_matrix_report["train"]["unprivileged"]["tn"] + confusion_matrix_report["train"]["unprivileged"]["fn"]
        confusion_matrix_report["test"]["all"]["pn"] = confusion_matrix_report["test"]["all"]["tn"] + confusion_matrix_report["test"]["all"]["fn"]
        confusion_matrix_report["test"]["privileged"]["pn"] = confusion_matrix_report["test"]["privileged"]["tn"] + confusion_matrix_report["test"]["privileged"]["fn"]
        confusion_matrix_report["test"]["unprivileged"]["pn"] = confusion_matrix_report["test"]["unprivileged"]["tn"] + confusion_matrix_report["test"]["unprivileged"]["fn"]

        dp_report = {
            'train_demographic_parity_ratio': demographic_parity_ratio(
                self._Ytrain, train_pred, sensitive_features=self._Xtrain[:, self._sfi]),
            'test_demographic_parity_ratio': demographic_parity_ratio(
                self._Ytest, test_pred, sensitive_features=self._Xtest[:, self._sfi]),
            'train_demographic_parity_difference': demographic_parity_difference(
                self._Ytrain, train_pred, sensitive_features=self._Xtrain[:, self._sfi]),
            'test_demographic_parity_difference': demographic_parity_difference(
                self._Ytest, test_pred, sensitive_features=self._Xtest[:, self._sfi]),

            'train_equalized_odds_ratio': equalized_odds_ratio(
                self._Ytrain, train_pred, sensitive_features=self._Xtrain[:, self._sfi]),
            'test_equalized_odds_ratio': equalized_odds_ratio(
                self._Ytest, test_pred, sensitive_features=self._Xtest[:, self._sfi]),
            'train_equalized_odds_difference': equalized_odds_difference(
                self._Ytrain, train_pred, sensitive_features=self._Xtrain[:, self._sfi]),
            'test_equalized_odds_difference': equalized_odds_difference(
                self._Ytest, test_pred, sensitive_features=self._Xtest[:, self._sfi]),
        }

        return {
            **train_report, 
            **test_report,
            "confusion_matrix": confusion_matrix_report,
            "training_sat": training_axioms_sat_report, 
            "sat": axiom_sat_report, 
            "fairness_metrics": dp_report
            }
