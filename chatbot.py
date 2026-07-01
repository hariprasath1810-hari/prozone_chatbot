import numpy as np
import joblib
from gensim.models import Word2Vec
import pandas as pd
import re, string

# Load trained model and embeddings
svm = joblib.load("models/chatbot_svm_model.pkl")
w2v = Word2Vec.load("models/chatbot_word2vec.model")

# Load your datasets
faq_df = pd.read_csv(r"C:\chatbot\dataset\new faq_csv (3).csv")
store_df = pd.read_csv(r"C:\chatbot\dataset\creating_dataset_csv_formate.csv")

all_df = pd.concat([faq_df, store_df], ignore_index=True)

# Clean + vector functions
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def get_sentence_vector(sentence):
    words = sentence.split()
    word_vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    if not word_vecs:
        return np.zeros(100)
    return np.mean(word_vecs, axis=0)

# Main chatbot logic
def chatbot_response(user_query):
    cleaned = clean_text(user_query)
    vec = get_sentence_vector(cleaned).reshape(1, -1)

    probs = svm.predict_proba(vec)[0]
    max_prob = np.max(probs)
    intent = svm.classes_[np.argmax(probs)]

    if max_prob < 0.5:  # threshold for unknown questions
        return "Sorry, I don't know about that."

    ans = all_df[all_df['intent'] == intent].iloc[0]['answer']
    return ans