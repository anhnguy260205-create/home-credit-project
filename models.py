"""Train and compare Logistic Regression, Decision Tree and Random Forest on German Credit.

Input:  german_credit_clean.csv (from cleaning_data.py)
Output: results/ folder with the comparison table, ROC curves, feature importance,
        confusion matrices and a drawing of the decision tree
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

INPUT_FILE = 'german_credit_clean.csv'
RESULTS_DIR = 'results'
SEED = 42

NUMERIC = ['age', 'log_credit_amount', 'duration']
CATEGORICAL = ['sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'purpose']

# Cost matrix from the original dataset: approving a bad applicant costs 5,
# rejecting a good applicant costs 1.
COST_BAD_APPROVED = 5
COST_GOOD_REJECTED = 1


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df['log_credit_amount'] = np.log(df['credit_amount'])  # reduce right skew
    X = df[NUMERIC + CATEGORICAL]
    y = df['risk_good']  # 1 = repays, 0 = defaults
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL),
    ])


def make_models():
    """Each model with the hyperparameters to try (tuned by 5-fold CV on the training set)."""
    return {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000),
            {'clf__C': [0.01, 0.1, 1, 10]},
        ),
        'Decision Tree': (
            DecisionTreeClassifier(random_state=SEED),
            {'clf__max_depth': [3, 4, 5, 6, 8],
             'clf__min_samples_leaf': [5, 10, 20, 40]},
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
            {'clf__max_depth': [4, 6, 8, None],
             'clf__min_samples_leaf': [1, 5, 10],
             'clf__max_features': ['sqrt', 0.5]},
        ),
    }


def evaluate(y_true, prob_good, threshold=0.5):
    pred = (prob_good >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()  # positive class = good
    return {
        'ROC AUC': roc_auc_score(y_true, prob_good),
        'Accuracy': accuracy_score(y_true, pred),
        'Precision (good)': precision_score(y_true, pred),
        'Recall (good)': recall_score(y_true, pred),
        'F1 (good)': f1_score(y_true, pred),
        'Bad caught': tn / (tn + fp),  # share of defaulters correctly rejected
        'Cost': COST_BAD_APPROVED * fp + COST_GOOD_REJECTED * fn,
    }


def feature_importance(name, model):
    prep, clf = model.named_steps['prep'], model.named_steps['clf']
    names = [n.split('__', 1)[1] for n in prep.get_feature_names_out()]
    if name == 'Logistic Regression':
        # Positive coefficient = raises the probability of repaying
        return pd.Series(clf.coef_[0], index=names).sort_values(key=abs, ascending=False)
    return pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)


def plot_roc(y_test, probs):
    plt.figure(figsize=(7, 6))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, prob):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess (AUC = 0.500)')
    plt.xlabel('False positive rate (bad applicants approved)')
    plt.ylabel('True positive rate (good applicants approved)')
    plt.title('ROC curves on the test set')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=150)
    plt.close()


def plot_confusion_matrices(y_test, probs):
    fig, axes = plt.subplots(1, len(probs), figsize=(5 * len(probs), 4.5))
    for ax, (name, prob) in zip(axes, probs.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, (prob >= 0.5).astype(int), display_labels=['bad', 'good'],
            cmap='Blues', colorbar=False, ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=150)
    plt.close()


def plot_importances(importances):
    fig, axes = plt.subplots(1, len(importances), figsize=(6 * len(importances), 6))
    for ax, (name, imp) in zip(axes, importances.items()):
        top = imp.head(10)[::-1]
        colors = ['tab:green' if v > 0 else 'tab:red' for v in top] \
            if name == 'Logistic Regression' else 'tab:blue'
        ax.barh(top.index, top.values, color=colors)
        ax.set_title(name + (' (coefficients)' if name == 'Logistic Regression' else ' (importance)'))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close()


def plot_decision_tree(model):
    names = [n.split('__', 1)[1] for n in model.named_steps['prep'].get_feature_names_out()]
    plt.figure(figsize=(24, 10))
    plot_tree(model.named_steps['clf'], feature_names=names, class_names=['bad', 'good'],
              filled=True, rounded=True, max_depth=3, fontsize=9, proportion=True)
    plt.title('Decision tree (top 3 levels)')
    plt.savefig(os.path.join(RESULTS_DIR, 'decision_tree.png'), dpi=120, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    print(f'Train: {len(X_train)} rows, Test: {len(X_test)} rows '
          f'({y_train.mean():.0%} good in both)\n')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, probs, importances, fitted = [], {}, {}, {}

    for name, (clf, grid) in make_models().items():
        pipe = Pipeline([('prep', make_preprocessor()), ('clf', clf)])
        search = GridSearchCV(pipe, grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best = search.best_index_
        cv_folds = [search.cv_results_[f'split{i}_test_score'][best] for i in range(5)]

        prob = model.predict_proba(X_test)[:, 1]  # probability of repaying
        probs[name], fitted[name] = prob, model
        importances[name] = feature_importance(name, model)

        params = {k.replace('clf__', ''): v for k, v in search.best_params_.items()}
        print(f'{name}: best params {params}')
        print(f'  CV AUC (train, 5-fold): {np.mean(cv_folds):.3f} +/- {np.std(cv_folds):.3f}')
        rows.append({'Model': name, 'CV AUC': np.mean(cv_folds), 'CV AUC std': np.std(cv_folds),
                     **evaluate(y_test, prob)})

    table = pd.DataFrame(rows).set_index('Model')
    table.round(3).to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'))
    print('\nTEST SET RESULTS (threshold 0.5, positive class = good)')
    print(table.round(3).to_string())
    print(f'\nCost = {COST_BAD_APPROVED} x bad approved + {COST_GOOD_REJECTED} x good rejected '
          f'(lower is better)')

    for name, imp in importances.items():
        imp.round(4).to_csv(os.path.join(RESULTS_DIR, f'importance_{name.lower().replace(" ", "_")}.csv'),
                            header=['value'])
        print(f'\nTop 5 features - {name}:')
        print(imp.head(5).round(3).to_string())

    plot_roc(y_test, probs)
    plot_confusion_matrices(y_test, probs)
    plot_importances(importances)
    plot_decision_tree(fitted['Decision Tree'])
    print(f'\nSaved results to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
