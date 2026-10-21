"""Train a LightGBM model on the Home Credit Default Risk data.

Usage:  python train.py
Output: submission.csv (Kaggle submission), model.txt (trained model),
        feature_importance.csv, validation_report.txt
"""
import gc
import os

import kagglehub
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
SEED = 42

DATA_DIR = kagglehub.competition_download('home-credit-default-risk')


def read(name):
    return pd.read_csv(os.path.join(DATA_DIR, f'{name}.csv'))


def one_hot(df):
    cats = [c for c in df.columns if df[c].dtype == 'object']
    df = pd.get_dummies(df, columns=cats, dummy_na=True, dtype=np.uint8)
    return df, [c for c in df.columns if any(c.startswith(cat + '_') for cat in cats)]


def aggregate(df, group_col, prefix):
    """Mean/max/min/sum of numeric columns and mean of one-hot columns, per group."""
    df, cat_cols = one_hot(df)
    num_cols = [c for c in df.columns if c not in cat_cols and c != group_col]
    aggs = {c: ['mean', 'max', 'min', 'sum'] for c in num_cols}
    aggs.update({c: ['mean'] for c in cat_cols})
    out = df.groupby(group_col).agg(aggs)
    out.columns = [f'{prefix}_{col}_{stat}'.upper() for col, stat in out.columns]
    out[f'{prefix}_COUNT'.upper()] = df.groupby(group_col).size()
    return out


def application():
    train, test = read('application_train'), read('application_test')
    df = pd.concat([train, test], ignore_index=True)
    df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['EMPLOYED_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    ext = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    df['EXT_SOURCE_MEAN'] = ext.mean(axis=1)
    df['EXT_SOURCE_PROD'] = ext.prod(axis=1, min_count=3)
    df['EXT_SOURCE_STD'] = ext.std(axis=1)

    df, _ = one_hot(df)
    return df


def bureau():
    b = read('bureau')
    bb = read('bureau_balance')
    bb_agg = aggregate(bb, 'SK_ID_BUREAU', 'BB')
    del bb
    b = b.join(bb_agg, on='SK_ID_BUREAU').drop(columns='SK_ID_BUREAU')
    del bb_agg
    gc.collect()
    out = aggregate(b, 'SK_ID_CURR', 'BUREAU')
    active = b[b['CREDIT_ACTIVE'] == 'Active']
    out = out.join(aggregate(active.select_dtypes('number'), 'SK_ID_CURR', 'BUREAU_ACTIVE'))
    return out


def previous_applications():
    prev = read('previous_application')
    for col in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
                'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        prev[col] = prev[col].replace(365243, np.nan)
    prev['APP_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT'].replace(0, np.nan)
    return aggregate(prev.drop(columns='SK_ID_PREV'), 'SK_ID_CURR', 'PREV')


def installments():
    ins = read('installments_payments')
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DAYS_PAST_DUE'] = (ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']).clip(lower=0)
    ins['DAYS_BEFORE_DUE'] = (ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']).clip(lower=0)
    return aggregate(ins.drop(columns='SK_ID_PREV'), 'SK_ID_CURR', 'INSTAL')


def pos_cash():
    return aggregate(read('POS_CASH_balance').drop(columns='SK_ID_PREV'), 'SK_ID_CURR', 'POS')


def credit_card():
    return aggregate(read('credit_card_balance').drop(columns='SK_ID_PREV'), 'SK_ID_CURR', 'CC')


def build_features():
    df = application()
    for name, fn in [('bureau', bureau), ('previous_application', previous_applications),
                     ('installments_payments', installments), ('POS_CASH_balance', pos_cash),
                     ('credit_card_balance', credit_card)]:
        print(f'Adding {name} features...')
        df = df.join(fn(), on='SK_ID_CURR')
        gc.collect()
    df = df.replace([np.inf, -np.inf], np.nan)
    df.columns = [''.join(ch if ch.isalnum() else '_' for ch in c) for c in df.columns]
    return df


def validation_report(y, prob, fold_aucs):
    """Score the out-of-fold predictions, i.e. each row predicted by a model that never saw it."""
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_threshold = thresholds[np.argmax(f1[:-1])]

    auc = roc_auc_score(y, prob)
    lines = [
        'VALIDATION REPORT (5-fold out-of-fold predictions)',
        '=' * 52,
        f'Rows: {len(y):,}   Defaults (TARGET=1): {int(y.sum()):,} ({y.mean():.1%})',
        '',
        f'ROC AUC        : {auc:.4f}   (competition metric; 0.5 = random, 1.0 = perfect)',
        f'AUC per fold   : {", ".join(f"{a:.4f}" for a in fold_aucs)}   (std {np.std(fold_aucs):.4f})',
        f'Gini           : {2 * auc - 1:.4f}   (= 2*AUC - 1, common in credit scoring)',
        f'PR AUC         : {average_precision_score(y, prob):.4f}   (baseline = default rate {y.mean():.4f})',
        f'Log loss       : {log_loss(y, prob):.4f}',
    ]
    for name, t in [('0.50 (default)', 0.5), (f'{best_threshold:.3f} (best F1)', best_threshold)]:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        lines += [
            '',
            f'At threshold {name}:',
            f'  Accuracy  : {accuracy_score(y, pred):.4f}',
            f'  Precision : {precision_score(y, pred, zero_division=0):.4f}   (of predicted defaults, share that defaulted)',
            f'  Recall    : {recall_score(y, pred):.4f}   (of actual defaults, share caught)',
            f'  F1        : {f1_score(y, pred):.4f}',
            f'  Confusion matrix:  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}',
        ]
    report = '\n'.join(lines)
    print('\n' + report)
    with open('validation_report.txt', 'w') as f:
        f.write(report + '\n')


def train(df):
    train_df = df[df['TARGET'].notna()]
    test_df = df[df['TARGET'].isna()]
    features = [c for c in df.columns if c not in ('TARGET', 'SK_ID_CURR')]
    X, y = train_df[features], train_df['TARGET']
    print(f'Train shape: {X.shape}, test shape: {test_df.shape}')

    params = dict(
        objective='binary', learning_rate=0.02, n_estimators=10000, num_leaves=34,
        colsample_bytree=0.3, subsample=0.87, subsample_freq=1, max_depth=8,
        reg_alpha=0.04, reg_lambda=0.07, min_split_gain=0.02, min_child_weight=40,
        random_state=SEED, verbose=-1,
    )

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(test_df))
    importance = pd.Series(0.0, index=features)
    fold_aucs = []
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(folds.split(X, y), 1):
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(test_df[features])[:, 1] / N_FOLDS
        importance += pd.Series(model.booster_.feature_importance('gain'), index=features)
        fold_aucs.append(roc_auc_score(y.iloc[va], oof[va]))
        print(f'Fold {fold} AUC: {fold_aucs[-1]:.4f}')

    validation_report(y, oof, fold_aucs)
    model.booster_.save_model('model.txt')
    pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'].astype(int), 'TARGET': test_pred}) \
        .to_csv('submission.csv', index=False)
    importance.sort_values(ascending=False).to_csv('feature_importance.csv', header=['gain'])
    print('Saved submission.csv, model.txt, feature_importance.csv, validation_report.txt')


if __name__ == '__main__':
    train(build_features())
