
import warnings
import os
import logging
import sys
import statistics

logging.basicConfig(level=logging.ERROR)
sys.path.append("../")
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from aif360.datasets import AdultDataset, CompasDataset, GermanDataset
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from tqdm import tqdm

import wandb
from utils import StandardScaleData_ExcludingFeature, set_seed


def main(dataset, feature):
    os.environ["WANDB_SILENT"] = "true"

    set_seed(42)

    project = "Tests"
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

    protected_attributes = { attributes['protected_attribute_names'][k] : {
        'name': attributes['protected_attribute_names'][k],
        'index': attributes['feature_names'].index(attributes['protected_attribute_names'][k]),
        'privileged': attributes['privileged_protected_attributes'][0][0],
        'unprivileged': attributes['unprivileged_protected_attributes'][0][0],
        'maps': metadata['protected_attribute_maps'][k]
    }
    for k, _ in enumerate(attributes['protected_attribute_names']) }

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

    wandb_hp['trainset_demographic_parity_difference'] = statistics.mean([demographic_parity_ratio(
        Ytrain, Ytrain, sensitive_features=Xtrain[:, idx]) for Xtrain, _, Ytrain, _ in sss_splits])
    wandb_hp['testset_demographic_parity_difference'] = statistics.mean([demographic_parity_ratio(
        Ytest, Ytest, sensitive_features=Xtest[:, idx]) for _, Xtest, _, Ytest in sss_splits])

    for i, (Xtrain, Xtest, Ytrain, Ytest) in tqdm(enumerate(sss_splits)):

        Xtrain, Xtest, scaler = StandardScaleData_ExcludingFeature(
            Xtrain, Xtest, protected_attributes[wandb_hp['sensitive_feature']]['index'])

        wandb_init = dict(
            project=project,
            name=f"svm",
            entity="albezjelt",
            config={
                **wandb_hp,},
            reinit=True
        )

        with wandb.init(**wandb_init) as run:
            svm = SVC(gamma='auto')
            svm.fit(Xtrain, Ytrain)
            Ytrain_pred = svm.predict(Xtrain)
            Ytest_pred = svm.predict(Xtest)

            train_report = {
                'train_classification_metrics': classification_report(
                    Ytrain, Ytrain_pred, output_dict=True
                )
            }

            test_report = {
                'test_classification_metrics': classification_report(
                    Ytest, Ytest_pred, output_dict=True
                )
            }

            dp_report = {
                'train_demographic_parity_difference': demographic_parity_difference(Ytrain, Ytrain_pred, sensitive_features=Xtrain[:, idx]),
                'test_demographic_parity_difference': demographic_parity_difference(Ytest, Ytest_pred, sensitive_features=Xtest[:, idx]),
                'train_demographic_parity_ratio': demographic_parity_ratio(Ytrain, Ytrain_pred, sensitive_features=Xtrain[:, idx]),
                'test_demographic_parity_ratio': demographic_parity_ratio(Ytest, Ytest_pred, sensitive_features=Xtest[:, idx]),
            }

            run.log({**train_report, **test_report, **dp_report})

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])