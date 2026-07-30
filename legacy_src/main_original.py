import sys

sys.path.append("../")
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from sklearn.model_selection import train_test_split
from utils import StandardScaleData_ExcludingFeature, LTNOps, set_seed, get_implies_operator
import KnowledgeBase
import tensorflow as tf
import numpy as np
from tqdm import trange

set_seed(42)


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
    #privilieged_value = ""
    #unprivileged_value = ""
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
    pbar = trange(5000, desc="Training", ncols=100)
    for epoch in pbar:
        with tf.GradientTape() as tape:
            loss = 1. - kb.train_step()  # type: ignore
        grads = tape.gradient(loss, kb.trainable_variables)
        optimizer.apply_gradients(zip(grads, kb.trainable_variables))

        if (epoch+1) % 1000 == 0 or epoch == 0:
            logs = kb.get_logs()
            train_acc = logs.get('train_classification_metrics', {}).get('accuracy')
            test_acc = logs.get('test_classification_metrics', {}).get('accuracy')
            pbar.set_postfix({
            'train_acc': f"{train_acc:.3f}",
            'test_acc':  f"{test_acc:.3f}"
            })


if __name__ == "__main__":
    main()