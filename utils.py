from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def top_n_similar(query_vec, matrix, n=1) -> List[Tuple[int, float]]:
    """Return indices and scores of top-n similar rows to query_vec inside matrix."""
    sims = cosine_similarity(query_vec, matrix)[0]
    order = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in order[:n]]

def ensure_columns(df: pd.DataFrame, required: list):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Present columns: {list(df.columns)}")




