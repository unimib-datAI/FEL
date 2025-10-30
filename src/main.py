import sys

sys.path.append("../")
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from sklearn.model_selection import train_test_split
from utils import StandardScaleData_ExcludingFeature, LTNOps, set_seed
import ltn
import KnowledgeBase
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import plotly.express as px
import pandas as pd

set_seed(42)


def get_implies_operator(implies):
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
    return impliesOperator



# qui server solo X, Y, e la sensitive feature
def get_training_dataset(dataset_orig, protected_attributes, sensitive_feature):
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_orig.features, 
        np.squeeze(dataset_orig.labels),
        test_size=0.33,
        stratify=np.squeeze(dataset_orig.labels))

    X_train, X_test, _ = StandardScaleData_ExcludingFeature(
            X_train, X_test, protected_attributes[sensitive_feature]['index'])
    
    return X_train, X_test, y_train, y_test



def main():

    dataset = 'compas'
    sensitive_feature = 'sex'
    implies = 'Godel'
    p_mean = 1
    aggregator_deviation = 2

    impliesOperator = get_implies_operator(implies)

    dataset_orig = CompasDataset()
    df, attributes = dataset_orig.convert_to_dataframe()
    metadata = dataset_orig.metadata


    # con l'input utente dobbiamo riuscire a creare questo dizionario
    protected_attributes = { attributes['protected_attribute_names'][k] : {
        'name': attributes['protected_attribute_names'][k],
        'index': attributes['feature_names'].index(attributes['protected_attribute_names'][k]),
        'privileged': attributes['privileged_protected_attributes'][0][0],
        'unprivileged': attributes['unprivileged_protected_attributes'][0][0],
        'maps': metadata['protected_attribute_maps'][k]  # mi sa che non serve
    }
    for k, _ in enumerate(attributes['protected_attribute_names']) }

    # e questo
    label_map = {
        "positive": [*metadata['label_maps'][0]][0],
        "negative": [*metadata['label_maps'][0]][1]
    }


    X_train, X_test, y_train, y_test  = get_training_dataset(dataset_orig, protected_attributes, sensitive_feature)


    ltnOps = LTNOps(impliesOperator, p_mean, aggregator_deviation)
    kb = KnowledgeBase.KnowledgeBase(
            X_train, X_test,
            y_train, y_test,
            label_map,
            protected_attributes[sensitive_feature]['privileged'],
            protected_attributes[sensitive_feature]['unprivileged'],
            hidden_layer_sizes=(50, 50),
            fuzzy_ops=ltnOps,
            sensitive_feature_index=protected_attributes[sensitive_feature]['index'],
            config_file='./src/KnowledgeBaseAxioms.json'
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in tqdm(range(5000)):
            with tf.GradientTape() as tape:
                loss = 1. - kb.train_step()  # type: ignore
            grads = tape.gradient(loss, kb.trainable_variables)
            optimizer.apply_gradients(zip(grads, kb.trainable_variables))



if __name__ == "__main__":
    main()