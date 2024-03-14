import xgboost
import lightgbm
import catboost
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import average_precision_score, matthews_corrcoef, fbeta_score, make_scorer

def impute_score(imputer, X_missing, y_missing):
    score_xgb = cross_validate(Pipeline([("imputer", imputer), 
                                         ("classifier", classifier_xgb)]), 
                               X_missing, y_missing, scoring={"MCC" : make_scorer(matthews_corrcoef), 
                                                              "PRC-AUC (AP)": "average_precision", 
                                                              "ROC-AUC": "roc_auc", 
                                                              "F-2 Score": make_scorer(fbeta_score, beta=2), 
                                                              "F-1.5 Score": make_scorer(fbeta_score, beta=1.5), 
                                                              "F-1 Score": "f1"}, cv=3, n_jobs=-1, verbose=1000, return_train_score=True)
    score_lgbm = cross_validate(Pipeline([("imputer", imputer), 
                                          ("classifier", classifier_lgbm)]), 
                                X_missing, y_missing, scoring={"MCC" : make_scorer(matthews_corrcoef), 
                                                               "PRC-AUC (AP)": "average_precision", 
                                                               "ROC-AUC": "roc_auc", 
                                                               "F-2 Score": make_scorer(fbeta_score, beta=2), 
                                                               "F-1.5 Score": make_scorer(fbeta_score, beta=1.5), 
                                                               "F-1 Score": "f1"}, cv=3, n_jobs=-1, verbose=1000, return_train_score=True)
    score_cat = cross_validate(Pipeline([("imputer", imputer), 
                                         ("classifier", classifier_cat)]), 
                               X_missing, y_missing, scoring={"MCC" : make_scorer(matthews_corrcoef), 
                                                              "PRC-AUC (AP)": "average_precision", 
                                                              "ROC-AUC": "roc_auc", 
                                                              "F-2 Score": make_scorer(fbeta_score, beta=2), 
                                                              "F-1.5 Score": make_scorer(fbeta_score, beta=1.5), 
                                                              "F-1 Score": "f1"}, cv=3, n_jobs=-1, verbose=1000, return_train_score=True)

    df_xgb = pd.DataFrame.from_dict({"train": [value.mean().round(4) for key, value in score_xgb.items() if "train" in key] + [score_xgb.get("fit_time").mean().round()], 
                                     "test": [value.mean().round(4) for key, value in score_xgb.items() if "test" in key] + [score_xgb.get("score_time").mean().round()]}).set_axis(["MCC", "PRC-AUC (AP)", "ROC-AUC", "F-2 Score", "F-1.5 Score", "F-1 Score", "fit_time, score_time"], axis=0)
    df_lgbm = pd.DataFrame.from_dict({"train": [value.mean().round(4) for key, value in score_lgbm.items() if "train" in key] + [score_lgbm.get("fit_time").mean().round()], 
                                      "test": [value.mean().round(4) for key, value in score_lgbm.items() if "test" in key] + [score_lgbm.get("score_time").mean().round()]}).set_axis(["MCC", "PRC-AUC (AP)", "ROC-AUC", "F-2 Score", "F-1.5 Score", "F-1 Score", "fit_time, score_time"], axis=0)
    df_cat = pd.DataFrame.from_dict({"train": [value.mean().round(4) for key, value in score_cat.items() if "train" in key] + [score_cat.get("fit_time").mean().round()], 
                                     "test": [value.mean().round(4) for key, value in score_cat.items() if "test" in key] + [score_cat.get("score_time").mean().round()]}).set_axis(["MCC", "PRC-AUC (AP)", "ROC-AUC", "F-2 Score", "F-1.5 Score", "F-1 Score", "fit_time, score_time"], axis=0)
    
    df_xgb.columns = [["XGBClassifier", "XGBClassifier"], ["train", "test"]]
    df_lgbm.columns = [["LGBMClassifier", "LGBMClassifier"], ["train", "test"]]
    df_cat.columns = [["CatBoostClassifier", "CatBoostClassifier"], ["train", "test"]]
    return pd.concat((df_xgb, df_lgbm, df_cat), axis=1)


classifier_xgb = xgboost.XGBClassifier(verbosity=3, tree_method="hist", n_jobs=-1, random_state=6, eval_metric=average_precision_score)
classifier_lgbm = lightgbm.LGBMClassifier(random_state=6, n_jobs=-1, verbosity=2, metric="average_precision")
classifier_cat = catboost.CatBoostClassifier(verbose=3, eval_metric="PRAUC", random_state=6)



import time
import xgboost
import catboost
import lightgbm
import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.notebook import tqdm
from imblearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer, recall_score, accuracy_score, roc_auc_score, precision_score, matthews_corrcoef, average_precision_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

class GetScore:
    def __init__(self, X_train, y_train, X_val, y_val, preprocess, resample=None, random_state=6):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_val = X_val.copy()
        self.y_val = y_val.copy()
        self.preprocess = preprocess
        self.resample = resample
        self.random_state = random_state
        self.model_info = {"dummy": {"name": "DummyClassifier", 
                                     "model": DummyClassifier(random_state=self.random_state)}, 
                           "sgd_svm": {"name": "SGDClassifier_SVM", 
                                       "model": SGDClassifier(n_jobs=-1, random_state=self.random_state)}, 
                           "sgd_lr": {"name": "SGDClassifier_LR", 
                                      "model": SGDClassifier(loss="log_loss", n_jobs=-1, random_state=self.random_state)}, 
                           "ridge": {"name": "RidgeClassifier", 
                                     "model": RidgeClassifier(random_state=self.random_state)}, 
                           "lr": {"name": "LogisticRegression", 
                                  "model": LogisticRegression(random_state=self.random_state, n_jobs=-1)}, 
                           "et": {"name": "ExtraTreeClassifier", 
                                  "model": ExtraTreeClassifier(random_state=self.random_state)}, 
                           "dt": {"name": "DecisionTreeClassifier", 
                                  "model": DecisionTreeClassifier(random_state=self.random_state)}, 
                           "ada": {"name": "AdaBoostClassifier", 
                                   "model": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, algorithm="SAMME", random_state=self.random_state)}, 
                           "ets": {"name": "ExtraTreesClassifier", 
                                   "model": ExtraTreesClassifier(n_jobs=-1, random_state=self.random_state)}, 
                           "rf": {"name": "RandomForestClassifier", 
                                  "model": RandomForestClassifier(n_jobs=-1, random_state=self.random_state)}, 
                           "gbc": {"name": "GradientBoostingClassifier", 
                                   "model": GradientBoostingClassifier(random_state=self.random_state)}, 
                           "hgbc": {"name": "HistGradientBoostingClassifier", 
                                    "model": HistGradientBoostingClassifier(random_state=self.random_state)}, 
                           "xgb": {"name": "XGBClassifier", 
                                   "model": xgboost.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state, n_jobs=-1, eval_metric="aucpr")}, 
                           "lgbm": {"name": "LGBMClassifier", 
                                    "model": lightgbm.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state, n_jobs=-1, metric="average_precision", verbosity=0)}, 
                           "cat": {"name": "CatBoostClassifier", 
                                   "model": catboost.CatBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state, thread_count=-1, eval_metric="PRAUC", verbose=0)}}
        self.mcc = list()
        self.ap = list()
        self.roc = list()
        self.f2 = list()
        self.f1_5 = list()
        self.f1 = list()
        self.precision = list()
        self.recall = list()
        self.accuracy = list()
        self.time = list()
        self.score_df = None

    def get_score(self, include=None, exclude=None):
        if self.mcc or self.ap or self.roc or self.f2 or self.f1_5 or self.f1 or self.precision or self.recall or self.accuracy or self.time or self.score_df:
            self.mcc = list()
            self.ap = list()
            self.roc = list()
            self.f2 = list()
            self.f1_5 = list()
            self.f1 = list()
            self.precision = list()
            self.recall = list()
            self.accuracy = list()
            self.time = list()
            self.score_df = None

        if include and exclude:
            raise TypeError("include 파라미터가 사용중일때 exclude 파라미터는 사용할 수 없습니다.")
        elif include:
            models = [model for id, model in self.model_info.items() if id in include]
        elif exclude:
            models = [model for id, model in self.model_info.items() if id not in exclude]
        else:
            models = [model for id, model in self.model_info.items()]

        with tqdm(models, desc="model - ", dynamic_ncols=True) as bar:
            for model_info in bar:
                bar.set_description(f"model - {model_info.get('name')} ")
                if self.resample:
                    full_pipeline = clone(Pipeline([("preprocess", self.preprocess), ("resample", self.resample), ("model", model_info.get("model"))], verbose=True))
                else:
                    full_pipeline = clone(Pipeline([("preprocess", self.preprocess), ("model", model_info.get("model"))], verbose=True))
                bar.set_postfix({"status" : "Training"})
                time_a = time.time()
                full_pipeline.fit(self.X_train, self.y_train)
                train_time = round(time.time() - time_a)
                bar.set_postfix({"status" : "Predicting"})
                time_b = time.time()
                y_train_pred, y_val_pred = full_pipeline.predict(self.X_train), full_pipeline.predict(self.X_val)
                predict_proba = True
                try:
                    y_train_score, y_val_score = full_pipeline.predict_proba(self.X_train)[:, 1], full_pipeline.predict_proba(self.X_val)[:, 1]
                except:
                    print(f"{model_info.get('name')}은(는) predict_proba를 지원하지 않습니다.")
                    predict_proba = False
                predict_time = round(time.time() - time_b)
                    
                bar.set_postfix({"status" : "Scoring"})
                self.mcc.extend([round(matthews_corrcoef(self.y_train, y_train_pred), 4), round(matthews_corrcoef(self.y_val, y_val_pred), 4)])
                self.ap.extend([round(average_precision_score(self.y_train, y_train_score), 4) if predict_proba else 0.0000, round(average_precision_score(self.y_val, y_val_score), 4) if predict_proba else 0.0000])
                self.roc.extend([round(roc_auc_score(self.y_train, y_train_score), 4) if predict_proba else 0.0000, round(roc_auc_score(self.y_val, y_val_score), 4) if predict_proba else 0.0000])
                self.f2.extend([round(fbeta_score(self.y_train, y_train_pred, beta=2), 4), round(fbeta_score(self.y_val, y_val_pred, beta=2), 4)])
                self.f1_5.extend([round(fbeta_score(self.y_train, y_train_pred, beta=1.5), 4), round(fbeta_score(self.y_val, y_val_pred, beta=1.5), 4)])
                self.f1.extend([round(fbeta_score(self.y_train, y_train_pred, beta=1), 4), round(fbeta_score(self.y_val, y_val_pred, beta=1), 4)])
                self.precision.extend([round(precision_score(self.y_train, y_train_pred), 4) if model_info.get("name") not in ["DummyClassifier", "AdaBoostClassifier"] else 0.0000, round(precision_score(self.y_val, y_val_pred), 4) if model_info.get("name") not in ["DummyClassifier", "AdaBoostClassifier"] else 0.0000])
                self.recall.extend([round(recall_score(self.y_train, y_train_pred), 4), round(recall_score(self.y_val, y_val_pred), 4)])
                self.accuracy.extend([round(accuracy_score(self.y_train, y_train_pred), 4), round(accuracy_score(self.y_val, y_val_pred), 4)])
                self.time.extend([train_time, predict_time])
                bar.set_description("model - ")
                bar.set_postfix()
            bar.set_description("done..!")

        self.score_df = pd.DataFrame(data=[self.mcc, self.ap, self.roc, self.f2, self.f1_5, self.f1, self.precision, self.recall, self.accuracy, self.time], 
                                     index=["MCC", "PRC-AUC (AP)", "ROC-AUC", "F2-Score", "F1.5-Score", "F1-Score", "Precision", "Recall", "Accuracy", "Train_time, Predict_time"], 
                                     columns=[np.array([[name, name] for name in [model_info.get("name") for model_info in models]]).flatten().tolist(), ["Train", "Test"] * len(models)])
        return self.score_df