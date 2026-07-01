# train_svm.py -- Train intent classifier + store embeddings using BERT + SVM
import os
import json
import pandas as pd
import numpy as np
from joblib import dump
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
import intents_rules

CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_auto(path):
    try:
        return pd.read_csv(path, encoding="cp1252")
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")

if __name__ == "__main__":
    cfg = load_config()
    models_dir = cfg["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------
    # 1) Load FAQ dataset
    # -------------------------------
    print("[train] Loading FAQ dataset...")
    faq = read_csv_auto(cfg["faq_csv"])

    if "Intent" not in faq.columns:
        print("[train] No 'Intent' column found, assigning with rules...")
        faq["Intent"] = faq["Question"].apply(intents_rules.get_intent)

    # Save processed FAQ with intents
    faq_cached_path = cfg["faq_cached_path"]
    faq.to_csv(faq_cached_path, index=False, encoding="utf-8-sig")
    print(f"[train] Cached FAQ with intents saved at {faq_cached_path}")

    # -------------------------------
    # 2) Load embedding model
    # -------------------------------
    embed_model_name = "all-MiniLM-L6-v2"
    print(f"[train] Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)

    # -------------------------------
    # 3) Encode FAQ questions
    # -------------------------------
    questions = faq["Question"].astype(str).fillna("").tolist()
    print(f"[train] Encoding {len(questions)} questions...")
    X = embedder.encode(questions, convert_to_numpy=True)

    # Save FAQ embeddings
    faq_emb_path = os.path.join(models_dir, "faq_embeddings.npy")
    np.save(faq_emb_path, X)
    print(f"[train] FAQ embeddings saved at {faq_emb_path}")

    # -------------------------------
    # 4) Encode labels
    # -------------------------------
    y = faq["Intent"].astype(str).tolist()
    LE = LabelEncoder()
    y_enc = LE.fit_transform(y)

    # -------------------------------
    # 5) Train/test split
    # -------------------------------
    counts = Counter(y_enc)
    if min(counts.values()) < 2:
        stratify = None
        print("[train] Some intents have <2 samples, skipping stratification.")
    else:
        stratify = y_enc

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=stratify
    )
    print(f"[train] Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # -------------------------------
    # 6) Train SVM
    # -------------------------------
    clf = SVC(kernel="linear", probability=True, C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    # -------------------------------
    # 7) Evaluate model
    # -------------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("[train] Accuracy:", round(acc, 3))

    unique_test_classes = np.unique(y_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=unique_test_classes,
        target_names=LE.inverse_transform(unique_test_classes)
    )
    print("[train] Classification Report:\n", report)

    # -------------------------------
    # 8) Save classifier + label encoder
    # -------------------------------
    clf_path = os.path.join(models_dir, "bert_svm_clf.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    dump(clf, clf_path)
    dump(LE, le_path)
    print(f"[train] Classifier saved at {clf_path}")
    print(f"[train] Label encoder saved at {le_path}")

    # -------------------------------
    # 9) Encode store knowledge
    # -------------------------------
    print("[train] Loading store knowledge...")
    try:
        store = read_csv_auto(cfg["knowledge_csv"])
    except Exception as e:
        print("[train] Could not load knowledge CSV:", e)
        store = pd.DataFrame()

    if not store.empty and "store_name" in store.columns:
        store_names = store["store_name"].astype(str).fillna("").tolist()
        if store_names:
            print(f"[train] Encoding {len(store_names)} store entries...")
            store_embs = embedder.encode(store_names, convert_to_numpy=True)
            store_emb_path = os.path.join(models_dir, "store_embeddings.npy")
            np.save(store_emb_path, store_embs)
            print(f"[train] Store embeddings saved at {store_emb_path}")

    # -------------------------------
    # 10) Save meta info
    # -------------------------------
    meta = {"embedding_model": embed_model_name}
    with open(os.path.join(models_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Meta info saved at {os.path.join(models_dir, 'meta.json')}")

    print("[train] ✅ Training with SVM finished successfully.")