
import logging
import os
import sys
import warnings
from tqdm import trange

logging.basicConfig(level=logging.ERROR)
sys.path.append("../")
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import statistics

import numpy as np
import tensorflow as tf
from aif360.datasets import AdultDataset, CompasDataset, GermanDataset
from fairlearn.metrics import (demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio)
import ltn
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import KnowledgeBase
import wandb
from utils import LTNOps, StandardScaleData_ExcludingFeature, set_seed

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# feature sensibile cittadianza, sex. genere ecc
# implies : tipo di im0licazione
# p_mean: parametro  operatori logici
# aggregator_deviation: 

def main(dataset, feature, implies, p_mean, aggregator_deviation):
    # Set seed for reproducibility
    set_seed(42)

    os.environ["WANDB_SILENT"] = "true"

    project = "Eq.odds"
    wandb_hp = {
        'dataset': dataset,
        'sensitive_feature': feature,
        'hidden_layer_sizes': (50, 50)
    }

    if dataset == 'adult':
        dataset_orig = AdultDataset()
    elif dataset == 'german':
        dataset_orig = GermanDataset()
    else:
        dataset_orig = CompasDataset()

    _, attributes = dataset_orig.convert_to_dataframe()
    metadata = dataset_orig.metadata

    # Protected attributes
    protected_attributes = {attributes['protected_attribute_names'][k]: {
        'name': attributes['protected_attribute_names'][k],
        'index': attributes['feature_names'].index(attributes['protected_attribute_names'][k]),
        'privileged': attributes['privileged_protected_attributes'][0][0],
        'unprivileged': attributes['unprivileged_protected_attributes'][0][0],
        'maps': metadata['protected_attribute_maps'][k]
    }
        for k, _ in enumerate(attributes['protected_attribute_names'])}
    label_map = {
        "positive": [*metadata['label_maps'][0]][0],
        "negative": [*metadata['label_maps'][0]][1]
    }

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
    sss_splits = []
    for train_index, test_index in sss.split(dataset_orig.features, np.squeeze(dataset_orig.labels)):
        Xtrain = dataset_orig.features[train_index]
        Xtest = dataset_orig.features[test_index]
        Ytrain = np.squeeze(dataset_orig.labels)[train_index]
        Ytest = np.squeeze(dataset_orig.labels)[test_index]
        sss_splits.append((Xtrain, Xtest, Ytrain, Ytest))

    idx = protected_attributes[wandb_hp['sensitive_feature']]['index']

    wandb_hp['trainset_demographic_parity_ratio'] = statistics.mean([demographic_parity_ratio(
        Ytrain, Ytrain, sensitive_features=Xtrain[:, idx]) for Xtrain, _, Ytrain, _ in sss_splits])
    wandb_hp['testset_demographic_parity_ratio'] = statistics.mean([demographic_parity_ratio(
        Ytest, Ytest, sensitive_features=Xtest[:, idx]) for _, Xtest, _, Ytest in sss_splits])

    wandb_hp['trainset_demographic_parity_difference'] = statistics.mean([demographic_parity_difference(
        Ytrain, Ytrain, sensitive_features=Xtrain[:, idx]) for Xtrain, _, Ytrain, _ in sss_splits])
    wandb_hp['testset_demographic_parity_difference'] = statistics.mean([demographic_parity_difference(
        Ytest, Ytest, sensitive_features=Xtest[:, idx]) for _, Xtest, _, Ytest in sss_splits])

    wandb_hp['trainset_equalized_odds_ratio'] = statistics.mean([equalized_odds_ratio(
        Ytrain, Ytrain, sensitive_features=Xtrain[:, idx]) for Xtrain, _, Ytrain, _ in sss_splits])
    wandb_hp['testset_equalized_odds_ratio'] = statistics.mean([equalized_odds_ratio(
        Ytest, Ytest, sensitive_features=Xtest[:, idx]) for _, Xtest, _, Ytest in sss_splits])

    wandb_hp['trainset_equalized_odds_difference'] = statistics.mean([equalized_odds_difference(
        Ytrain, Ytrain, sensitive_features=Xtrain[:, idx]) for Xtrain, _, Ytrain, _ in sss_splits])
    wandb_hp['testset_equalized_odds_difference'] = statistics.mean([equalized_odds_difference(
        Ytest, Ytest, sensitive_features=Xtest[:, idx]) for _, Xtest, _, Ytest in sss_splits])

    # Implies operator
    if implies == 'KleeneDienes':
        impliesOperator = ltn.fuzzy_ops.Implies_KleeneDienes()
    elif implies == 'Godel':
        impliesOperator = ltn.fuzzy_ops.Implies_Godel()
    elif implies == 'Reichenbach':
        impliesOperator = ltn.fuzzy_ops.Implies_Reichenbach()
    elif implies == 'Goguen':
        impliesOperator = ltn.fuzzy_ops.Implies_Goguen()
    elif implies == 'Luk':
        impliesOperator = ltn.fuzzy_ops.Implies_Luk()
    else:
        raise Exception(f'Implies operator {implies} undefined')

    fuzzy_ops = LTNOps(impliesOperator, int(p_mean), int(aggregator_deviation))

    for _, (Xtrain, Xtest, Ytrain, Ytest) in tqdm(enumerate(sss_splits), desc=f"{dataset}, {feature}"):

        Xtrain, Xtest, _ = StandardScaleData_ExcludingFeature(
            Xtrain, Xtest, protected_attributes[wandb_hp['sensitive_feature']]['index'])

        kb = KnowledgeBase.KnowledgeBase(
            Xtrain, Xtest,
            Ytrain, Ytest,
            label_map,
            protected_attributes[wandb_hp['sensitive_feature']]['privileged'],
            protected_attributes[wandb_hp['sensitive_feature']]['unprivileged'],
            hidden_layer_sizes=wandb_hp['hidden_layer_sizes'],
            fuzzy_ops=fuzzy_ops,
            sensitive_feature_index=protected_attributes[wandb_hp['sensitive_feature']]['index'],
            config_file='./src/KnowledgeBaseAxioms.json'
        )

        wandb_hp['learning_rate'] = 0.001
        optimizer = tf.keras.optimizers.Adam(learning_rate=wandb_hp['learning_rate'])
        wandb_hp['optimizer'] = optimizer.__class__
        wandb_hp['epochs'] = 1000

        wandb_init = dict(
            project=project,
            name=f"{' | '.join([ax['name'] for ax in kb.config if ax['infos']['training']]) }",
            entity="albezjelt",
            config={
                **wandb_hp,
                **kb.axioms,
                'weights': kb.weights,
                'config.json': kb.config,
                'protected_attributes': protected_attributes,
                'label_map': label_map,
                'fuzzy_ops': fuzzy_ops.get_description()},
            reinit=True
        )

        # Training loop
        with wandb.init(**wandb_init) as run:
            pbar = trange(wandb_hp['epochs'], desc="Training", ncols=100)
            for epoch in pbar:

                with tf.GradientTape() as tape:
                    loss = 1. - kb.train_step()  # type: ignore
                grads = tape.gradient(loss, kb.trainable_variables)
                optimizer.apply_gradients(zip(grads, kb.trainable_variables))

                if (epoch+1) % 10 == 0 or epoch == 0:
                    run.log({
                        'epochs': epoch+1,
                        **kb.get_logs()
                    })
                    logs = kb.get_logs()
                    train_acc = logs.get('train_classification_metrics', {}).get('accuracy')
                    test_acc = logs.get('test_classification_metrics', {}).get('accuracy')
                    pbar.set_postfix({
                    'train_acc': f"{train_acc:.3f}",
                    'test_acc':  f"{test_acc:.3f}"
                    })

            run.finish()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
