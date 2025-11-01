import sys

sys.path.append("../")
from sklearn.model_selection import train_test_split
from utils import StandardScaleData_ExcludingFeature, LTNOps, set_seed, get_implies_operator
import KnowledgeBase
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import trange
import yaml
import argparse
import os
import numpy as np
import tensorflow as tf

set_seed(42)


def save_kb_weights(kb, filepath: str):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    weights = {f"var_{i}": v.numpy() for i, v in enumerate(kb.trainable_variables)}
    np.savez(filepath, **weights)


def load_kb_weights(kb, filepath: str):
    data = np.load(filepath, allow_pickle=True)
    vars = kb.trainable_variables
    for i, v in enumerate(vars):
        key = f"var_{i}"
        if key not in data:
            raise KeyError(f"Missing weight '{key}' in {filepath}")
        v.assign(data[key])
    return kb



def simple_inference(kb, X, n=5):
    preds = kb._oracle.predict(X) 
    preds = np.asarray(preds)
    print("Sample predictions:", preds[:n])
    return preds


def prepare_dataset(df, target_variable,  protected_attribute):
    Y = df[target_variable].to_numpy()
    X = df.drop(columns=[target_variable]).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        np.squeeze(Y),
        test_size=0.33,
        stratify=np.squeeze(Y))

    X_train, X_test, _ = StandardScaleData_ExcludingFeature(
            X_train, X_test, protected_attribute['index'])
    
    return df, X_train, X_test, y_train, y_test


def main(args):
    with open(args.config, 'r') as f:
        conf =  yaml.safe_load(f)

    df = pd.read_csv(args.dataset)

    protected_attribute = {
        'name': conf["data"]["sensitive_feature"],
        'index': df.columns.get_loc(conf["data"]["sensitive_feature"]),
        'privileged': conf["data"]["protected_values"]["privileged"],
        'unprivileged': conf["data"]["protected_values"]["unprivileged"]
    }

    label_map = {
        "positive": conf["data"]["labels"]["positive"],
        "negative": conf["data"]["labels"]["negative"]
    }


    df, X_train, X_test, y_train, y_test  = prepare_dataset(df, conf["data"]["target_variable"],  protected_attribute)
    impliesOperator = get_implies_operator(conf["model"]["implies"])

    ltnOps = LTNOps(impliesOperator, conf["model"]["p_mean"], conf["model"]["aggregator_deviation"])
    kb = KnowledgeBase.KnowledgeBase(
            X_train, X_test,
            y_train, y_test,
            label_map,
            protected_attribute['privileged'],
            protected_attribute['unprivileged'],
            hidden_layer_sizes= conf["model"]["hidden_layer_sizes"],
            fuzzy_ops=ltnOps,
            sensitive_feature_index=protected_attribute['index'],
            config_file='./src/KnowledgeBaseAxioms.json'
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    pbar = trange(conf["training"]["epochs"], desc="Training", ncols=100)
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
    
    save_kb_weights(kb, "models/kb.npz")
    kb = load_kb_weights(kb, "models/kb.npz")
    simple_inference(kb, X_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./src/config.yaml', help='Path to config file')
    parser.add_argument('--dataset', default='./src/compas.csv', help='Optional override for dataset CSV path')
    args = parser.parse_args()
    main(args)