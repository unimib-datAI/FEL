import sys

sys.path.append("../")
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from sklearn.model_selection import train_test_split
from utils import StandardScaleData_ExcludingFeature, LTNOps, set_seed, get_implies_operator
import KnowledgeBase
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import trange

set_seed(42)


# deve ritornare X e Y
def read_dataset(csv, target_var):
    df = pd.read_csv(csv)
    Y = df[target_var].to_numpy()
    X = df.drop(columns=[target_var]).to_numpy()
    return df, X, Y


# qui server solo X, Y, e la sensitive feature
def get_training_dataset(X, Y, protected_attributes):
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        np.squeeze(Y),
        test_size=0.33,
        stratify=np.squeeze(Y))

    X_train, X_test, _ = StandardScaleData_ExcludingFeature(
            X_train, X_test, protected_attributes['index'])
    
    return X_train, X_test, y_train, y_test


def build_predicted_attributes_dict(dataset_df, sensitive_feature, privilieged_value, unprivileged_value):
    protected_attributes = {
        sensitive_feature: {
            'name': sensitive_feature,
            'index': dataset_df.columns.get_loc(sensitive_feature),
            'privileged': privilieged_value,
            'unprivileged': unprivileged_value,
            'maps': {}  # mi sa che non serve
        }
    }
    return protected_attributes

def main():
    dataset_path = "./src/compas.csv"
    sensitive_feature = 'sex'
    privilieged_value_for_sensitive_feature= 1
    unprivileged_value_for_sensitive_feature = 0
    target_variable  = "two_year_recid"
    positive_label = 1  # The positive value is what the model should try to balance fairly across groups.
    negative_label = 0  # The negative value is not directly balanced in fairness metrics.
    implies = 'Godel'
    p_mean = 1
    aggregator_deviation = 2
    hidden_layer_sizes=(50, 50)
    epochs = 5000
    

    df, X, Y = read_dataset(dataset_path, target_variable)
    impliesOperator = get_implies_operator(implies)


    # con l'input utente dobbiamo riuscire a creare questo dizionario
    protected_attributes = {
        'name': sensitive_feature,
        'index': df.columns.get_loc(sensitive_feature),
        'privileged': privilieged_value_for_sensitive_feature,
        'unprivileged': unprivileged_value_for_sensitive_feature
    }

    label_map = {
        "positive": positive_label,
        "negative": negative_label
    }


    X_train, X_test, y_train, y_test  = get_training_dataset(X, Y, protected_attributes)

    ltnOps = LTNOps(impliesOperator, p_mean, aggregator_deviation)
    kb = KnowledgeBase.KnowledgeBase(
            X_train, X_test,
            y_train, y_test,
            label_map,
            protected_attributes['privileged'],
            protected_attributes['unprivileged'],
            hidden_layer_sizes= hidden_layer_sizes,
            fuzzy_ops=ltnOps,
            sensitive_feature_index=protected_attributes['index'],
            config_file='./src/KnowledgeBaseAxioms.json'
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    pbar = trange(epochs, desc="Training", ncols=100)
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