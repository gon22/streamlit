import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelect(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.drop(["sub_grade", "emp_length", "addr_state", "mo_sin_rcnt_rev_tl_op"], axis=1)
        return X_copy

class LimitOutlier(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["dti"] = X_copy.dti.apply(lambda x: 0 if x <= 0 else x if x <= 40 else 40)
        X_copy["revol_util"] = X_copy.revol_util.apply(lambda x: 100 if x > 100 else x)
        X_copy["last_fico_range_high"] = np.where(X_copy.last_fico_range_low == 0, 304, X_copy.last_fico_range_high)
        X_copy["last_fico_range_low"] = np.where(X_copy.last_fico_range_low == 0, 300, X_copy.last_fico_range_low)
        return X_copy

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["term"] = X_copy.term.str.extract("(\d+)").astype(np.float64)
        X_copy["open_acc_rate"] = (X_copy.open_acc / X_copy.total_acc * 100).round(1)
        X_copy["last_fico_score"] = X_copy[["last_fico_range_low", "last_fico_range_high"]].mean(axis=1).round()
        X_copy["installment"] = np.vectorize(lambda x, y, z: round((x * y * 0.01 / 12) * ((1 + y * 0.01 / 12) ** z) / (((1 + y * 0.01 / 12) ** z) - 1), 2))(X_copy.loan_amnt, X_copy.int_rate, X_copy.term)
        X_copy = X_copy.drop(["open_acc", "total_acc", "last_fico_range_low", "last_fico_range_high"], axis=1)
        return X_copy