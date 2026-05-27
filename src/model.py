import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


def split_with_smote(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Stratified split then SMOTE on the training set only.
    Test set stays untouched so evaluation reflects real-world imbalance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f'[split] Train (after SMOTE) : {X_train_res.shape[0]:,} rows '
          f'| Test: {X_test.shape[0]:,} rows')
    print(f'[split] Resampled balance   : '
          f'{dict(pd.Series(y_train_res).value_counts())}')

    return X_train_res, X_test, y_train_res, y_test


def _get_candidate_models(random_state: int = 42) -> dict:
    # VotingClassifier combines LR + RF + GBM as a soft-voting ensemble
    lr  = LogisticRegression(max_iter=1000, C=0.5, random_state=random_state)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  random_state=random_state, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', random_state=random_state,
                        n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                      learning_rate=0.1, random_state=random_state)
    vote = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('gbm', gbm)],
        voting='soft', n_jobs=-1,
    )
    return {'XGBoost': xgb, 'Random Forest': rf,
            'Gradient Boosting': gbm, 'Voting Ensemble': vote}


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_test, y_pred), 4),
        'F1'       : round(f1_score(y_test, y_pred), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_prob), 4),
    }


def train_and_compare(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict, dict]:
    """Train all candidates with 5-fold CV and return a comparison table."""
    candidates = _get_candidate_models()
    rows       = []
    trained    = {}

    for name, clf in candidates.items():
        print(f'  ▶ {name} ...', end=' ', flush=True)
        clf.fit(X_train, y_train)
        trained[name] = clf

        row = _evaluate(clf, X_test, y_test, name)

        cv = cross_val_score(clf, X_train, y_train, cv=5,
                              scoring='roc_auc', n_jobs=-1)
        row['CV-AUC (mean)'] = round(cv.mean(), 4)
        row['CV-AUC (std)']  = round(cv.std(),  4)

        rows.append(row)
        print(f"ROC-AUC={row['ROC-AUC']}  CV={row['CV-AUC (mean)']:.4f}±{row['CV-AUC (std)']:.4f}")

    df_results = (
        pd.DataFrame(rows)
        .set_index('Model')
        .sort_values('ROC-AUC', ascending=False)
    )

    # compact format for the pkl bundle
    raw_results = {
        name: {'accuracy': r['Accuracy'], 'roc_auc': r['ROC-AUC']}
        for name, r in df_results.iterrows()
    }

    return df_results, trained, raw_results


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    random_state: int = 42,
) -> RandomForestClassifier:
    """RandomizedSearchCV over RF hyperparameters. Returns the best estimator."""
    param_dist = {
        'n_estimators'     : [100, 200, 300, 400, 500],
        'max_depth'        : [5, 8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'max_features'     : ['sqrt', 'log2', 0.5],
        'class_weight'     : [None, 'balanced'],
    }
    base   = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, scoring='roc_auc',
        cv=5, random_state=random_state, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f'[tune] Best CV ROC-AUC : {search.best_score_:.4f}')
    print(f'[tune] Best params     : {search.best_params_}')
    return search.best_estimator_


def plot_feature_importance(
    model, feature_names: list, top_n: int = 15, save_path: str = None
) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax.barh(np.array(feature_names)[idx], importances[idx], color=colors)
    ax.set_title(f'Top {top_n} Feature Importances — Random Forest',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(
    model, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None
) -> None:
    cm  = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — Best Model', fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curve(
    model, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color='#e63946')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2)
    ax.set_title('ROC Curve — Best Model', fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_model_bundle(
    best_model,
    all_models: dict,
    encoders: dict,
    scaler,
    feature_names: list,
    best_model_name: str,
    results: dict,
    save_path: str,
) -> None:
    """
    Save everything needed for inference to a single pkl file.
    Includes the model, encoders, scaler, feature list, and per-model metrics.
    all_models keeps every trained estimator for offline comparison.
    """
    bundle = {
        'model'          : best_model,
        'encoder'        : encoders,
        'scaler'         : scaler,
        'features'       : feature_names,
        'best_model_name': best_model_name,
        'results'        : results,
        'all_models'     : all_models,
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'[save] PKL bundle saved → {save_path}')


def load_model_bundle(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
