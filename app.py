# app.py -- Flask server using BERT embeddings + Logistic Regression classifier + Not Found fallback
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from joblib import load
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import intents_rules

CONFIG_PATH = "config.json"
MODELS_DIR = "models"

CLF_PATH = os.path.join(MODELS_DIR, "bert_svm_clf.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
FAQ_CACHED = os.path.join(MODELS_DIR, "faq_with_intents.csv")
FAQ_EMB = os.path.join(MODELS_DIR, "faq_embeddings.npy")
STORE_EMB_PATH = os.path.join(MODELS_DIR, "store_embeddings.npy")
FAQ_META_PATH = os.path.join(MODELS_DIR, "faq_meta.json")

app = Flask(__name__, static_folder="static")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_auto(path):
    try:
        return pd.read_csv(path, encoding="cp1252")
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")

cfg = load_config()

print("[app] Loading classifier and label encoder...")
CLF = load(CLF_PATH)
LE = load(LE_PATH)

print("[app] Loading FAQ and embeddings...")
FAQ = read_csv_auto(FAQ_CACHED)
faq_embeddings = np.load(FAQ_EMB)

faq_threshold = 0.5
if os.path.exists(FAQ_META_PATH):
    with open(FAQ_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        faq_threshold = meta.get("threshold", 0.5)

print("[app] Loading knowledge CSV...")
try:
    STORE = read_csv_auto(cfg["knowledge_csv"])
except Exception as e:
    print("[app] Could not load knowledge CSV:", e)
    STORE = pd.DataFrame()

print("[app] Loading SentenceTransformer model (this may take a moment)...")
meta_path = os.path.join(MODELS_DIR, "meta.json")
embed_model_name = "all-MiniLM-L6-v2"
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        embed_model_name = meta.get("embedding_model", embed_model_name)

embedder = SentenceTransformer(embed_model_name)

STORE_EMB = None
if os.path.exists(STORE_EMB_PATH):
    STORE_EMB = np.load(STORE_EMB_PATH)

@app.get("/")
def index():
    return send_from_directory("static", "index.html")

def resolve_store_by_embedding(user_text):
    if STORE_EMB is None or STORE.empty:
        return None, 0.0
    q_emb = embedder.encode([user_text], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, STORE_EMB)[0]
    idx = int(np.argmax(sims))
    return STORE.iloc[idx]["store_name"], float(sims[idx])

@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"reply": "Please type a question."})

    return process_message(user_text)

@app.get("/test")
def test():
    user_text = request.args.get("msg", "").strip()
    if not user_text:
        return jsonify({"error": "Please provide a query using ?msg=your+question"})
    return process_message(user_text)

# -----------------------------
# Core processing logic
# -----------------------------
def process_message(user_text):
    # 1) Encode
    q_emb = embedder.encode([user_text], convert_to_numpy=True)

    # ---- Removed Not Found logic ----
    # sims_all = cosine_similarity(q_emb, faq_embeddings)[0]
    # max_sim = float(np.max(sims_all))
    # if max_sim < faq_threshold:
    #     return jsonify({
    #         "reply": "Sorry, I don’t have information about that.",
    #         "intent": "not_found",
    #         "confidence": round(max_sim, 3)
    #     })

    # 2) Predict intent
    try:
        probs = CLF.predict_proba(q_emb)[0]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        pred_enc = CLF.predict(q_emb)[0]
        predicted_intent = LE.inverse_transform([pred_enc])[0]
    except Exception:
        pred_enc = CLF.predict(q_emb)[0]
        predicted_intent = LE.inverse_transform([pred_enc])[0]
        best_prob = 1.0

    # 3) If low confidence, try rule-based fallback
    CONF_THRESHOLD = 0.55
    if best_prob < CONF_THRESHOLD:
        predicted_intent = intents_rules.get_intent(user_text)

    # 4) Store query handling
    if predicted_intent == "store_info" and not STORE.empty:
        name, score = resolve_store_by_embedding(user_text)
        if name and score > 0.20:
            row = STORE[STORE["store_name"].str.contains(name, case=False, na=False)]
            if not row.empty:
                r = row.iloc[0].to_dict()
                loc = r.get("location") or r.get("floor") or "location not available"
                hours = r.get("timings") or r.get("hours") or "timings not available"
                phone = r.get("phone") or r.get("contact") or ""
                reply = f"{name} is at {loc}. Timings: {hours}."
                if phone:
                    reply += f" Phone: {phone}."
                return jsonify({
                    "reply": reply,
                    "intent": "store_info",
                    "store_match_score": score,
                    "confidence": best_prob
                })

    # 5) FAQ match inside predicted intent
    subset = FAQ[FAQ["Intent"] == predicted_intent]
    if subset.empty:
        subset = FAQ
    idxs = subset.index.to_numpy()
    sub_embs = faq_embeddings[idxs]
    sims = cosine_similarity(q_emb, sub_embs)[0]
    best_local = int(np.argmax(sims))
    best_global_idx = idxs[best_local]
    best_row = FAQ.iloc[best_global_idx]
    reply = best_row["Answer"]

    return jsonify({
        "reply": reply,
        "matched_question": best_row["Question"],
        "intent": predicted_intent,
        "confidence": best_prob
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

