"""Fold-local feature transformers with stable module paths for joblib/loky."""
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MONEY = re.compile(r"(?i)(?:(?:\b(?:PEN|USD)(?=\s*\d)|US\s*\$|S\s*/\.?|\$)\s*\d+(?:[.,]\d+)*|\b\d+(?:[.,]\d+)*\s*(?:soles?|d[oó]lares?)(?!\w))")
URL = re.compile(r"(?i)https?://\S+|www\.\S+")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


class ListingText(BaseEstimator, TransformerMixin):
    """Join title/description inside the fitted fold, preserving row order."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).fillna("").astype(str)
        joined = frame.iloc[:, 0].str.cat(frame.iloc[:, 1], sep=" ")
        return joined.map(lambda text: EMAIL.sub(" ", URL.sub(" ", MONEY.sub(" monetaryamount ", text)))).to_numpy()

    def audit(self, X):
        frame = pd.DataFrame(X).fillna("").astype(str)
        joined = frame.iloc[:, 0].str.cat(frame.iloc[:, 1], sep=" ")
        return {"rows": len(frame), "rows_with_explicit_currency_amount": int(joined.map(lambda x: bool(MONEY.search(x))).sum()),
                "explicit_currency_amount_matches": int(joined.map(lambda x: len(MONEY.findall(x))).sum()),
                "rows_with_url": int(joined.map(lambda x: bool(URL.search(x))).sum()),
                "rows_with_email": int(joined.map(lambda x: bool(EMAIL.search(x))).sum()),
                "interpretation": "Fixed currency expressions masked without consulting target; unmarked numbers remain, so absence of target echo is not established."}


class SafeTextSVD(BaseEstimator, TransformerMixin):
    """Train-only TF-IDF/SVD with a fixed-width, empty-vocabulary safeguard."""
    def __init__(self, n_components=4, min_df=3, max_features=1500, random_state=42):
        self.n_components = n_components
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                         ngram_range=(1, 2), min_df=self.min_df,
                                         max_df=.95, max_features=self.max_features,
                                         sublinear_tf=True)
        self.svd_ = None
        self.empty_ = False
        try:
            matrix = self.vectorizer_.fit_transform(X)
        except ValueError as exc:
            if not any(s in str(exc).lower() for s in ("empty vocabulary", "no terms remain", "max_df corresponds")):
                raise
            self.empty_ = True
            return self
        self.effective_components_ = min(self.n_components, matrix.shape[1])
        if matrix.shape[1] == 1:
            self.single_term_ = True
        else:
            self.single_term_ = False
            self.svd_ = TruncatedSVD(n_components=self.effective_components_, random_state=self.random_state)
            self.svd_.fit(matrix)
        return self

    def transform(self, X):
        output = np.zeros((len(X), self.n_components), dtype=float)
        if self.empty_:
            return output
        matrix = self.vectorizer_.transform(X)
        values = matrix.toarray() if self.single_term_ else self.svd_.transform(matrix)
        output[:, :values.shape[1]] = values
        return output

    def get_feature_names_out(self, input_features=None):
        return np.array([f"text_svd_{i+1}" for i in range(self.n_components)], dtype=object)



