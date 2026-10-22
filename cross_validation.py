"""Repeated cross-validation: are the differences between the 3 models real or luck?

Runs 10 x 5-fold stratified cross-validation (50 train/test splits) on all 1,000 applicants,
using the hyperparameters chosen in models.py, then compares the models pair by pair.

Input:  german_credit_clean.csv
Output: results/cv_scores.csv, results/cv_summary.csv, results/cv_auc_boxplot.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from models import (CATEGORICAL, COST_BAD_APPROVED, COST_GOOD_REJECTED, INPUT_FILE, NUMERIC,
                    RESULTS_DIR, SEED, make_preprocessor)

N_SPLITS = 5
N_REPEATS = 10

# Best hyperparameters found by GridSearchCV in models.py
MODELS = {
    'Logistic Regression': LogisticRegression(C=1, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=6, max_features=0.5,
                                            random_state=SEED, n_jobs=-1),
}


def bad_caught(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def cost(y_true, y_pred):
    _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return COST_BAD_APPROVED * fp + COST_GOOD_REJECTED * fn


SCORING = {
    'auc': 'roc_auc',
    'accuracy': 'accuracy',
    'f1': 'f1',
    'bad_caught': make_scorer(bad_caught),
    'cost': make_scorer(cost),
}


def corrected_ttest(a, b):
    """Paired t-test for repeated k-fold CV with the Nadeau & Bengio (2003) correction.

    The folds share training data, so a normal t-test is over-confident; the correction
    widens the variance by (1/k_total + n_test/n_train)."""
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    test_train_ratio = 1 / (N_SPLITS - 1)
    var = diff.var(ddof=1) * (1 / n + test_train_ratio)
    t = diff.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return diff.mean(), (diff > 0).mean(), p


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    df['log_credit_amount'] = np.log(df['credit_amount'])
    X, y = df[NUMERIC + CATEGORICAL], df['risk_good']

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    print(f'{N_REPEATS} x {N_SPLITS}-fold cross-validation = {N_SPLITS * N_REPEATS} splits '
          f'on {len(X)} applicants\n')

    scores = {}
    for name, clf in MODELS.items():
        pipe = Pipeline([('prep', make_preprocessor()), ('clf', clf)])
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        scores[name] = {m: res[f'test_{m}'] for m in SCORING}
        print(f'{name}: done')

    # Summary: mean, std and 95% range of each metric
    rows = []
    for name, s in scores.items():
        row = {'Model': name}
        for m, v in s.items():
            row[f'{m} mean'] = v.mean()
            row[f'{m} std'] = v.std()
        lo, hi = np.percentile(s['auc'], [2.5, 97.5])
        row['auc 95% range'] = f'{lo:.3f} - {hi:.3f}'
        rows.append(row)
    summary = pd.DataFrame(rows).set_index('Model')
    summary.round(3).to_csv(os.path.join(RESULTS_DIR, 'cv_summary.csv'))

    print('\nRESULTS (mean +/- std over 50 splits, threshold 0.5)')
    print(f'{"Model":<22}{"AUC":>18}{"Accuracy":>18}{"F1 (good)":>18}{"Bad caught":>18}{"Cost":>16}')
    for name, s in scores.items():
        cells = [f'{s[m].mean():.3f} +/- {s[m].std():.3f}' for m in ['auc', 'accuracy', 'f1', 'bad_caught']]
        cells.append(f'{s["cost"].mean():.1f} +/- {s["cost"].std():.1f}')
        print(f'{name:<22}' + ''.join(f'{c:>18}' for c in cells[:4]) + f'{cells[4]:>16}')
    print('(Cost is per 200 applicants: 5 x bad approved + 1 x good rejected; lower is better)')

    # Pairwise comparison on AUC (same splits for every model, so the test is paired)
    print('\nIS THE DIFFERENCE IN AUC REAL? (corrected paired t-test, p < 0.05 = significant)')
    names = list(MODELS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diff, win_rate, p = corrected_ttest(scores[a]['auc'], scores[b]['auc'])
            verdict = 'SIGNIFICANT' if p < 0.05 else 'not significant'
            print(f'  {a} vs {b}: diff {diff:+.3f}, {a} wins {win_rate:.0%} of splits, '
                  f'p = {p:.3f} -> {verdict}')

    pd.DataFrame({f'{name} {m}': s[m] for name, s in scores.items() for m in SCORING}) \
        .round(4).to_csv(os.path.join(RESULTS_DIR, 'cv_scores.csv'), index_label='split')

    plt.figure(figsize=(8, 5))
    plt.boxplot([scores[n]['auc'] for n in names], showmeans=True)
    plt.xticks(range(1, len(names) + 1), names)
    plt.ylabel('ROC AUC')
    plt.title(f'AUC over {N_REPEATS} x {N_SPLITS}-fold cross-validation ({N_SPLITS * N_REPEATS} splits)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cv_auc_boxplot.png'), dpi=150)
    plt.close()
    print(f'\nSaved cv_summary.csv, cv_scores.csv and cv_auc_boxplot.png to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
