"""Clean the German Credit dataset.

Input:  german_credit_data.csv
Output: german_credit_clean.csv
"""
import pandas as pd

INPUT_FILE = 'german_credit_data.csv'
OUTPUT_FILE = 'german_credit_clean.csv'

CATEGORY_COLS = ['sex', 'housing', 'saving_accounts',
                 'checking_account', 'purpose', 'risk']
NUMERIC_COLS = ['age', 'credit_amount', 'duration']
JOB_LABELS = {0: 'unskilled_nonresident',
              1: 'unskilled_resident', 2: 'skilled', 3: 'highly_skilled'}


def load_data():
    """Step 1: read the CSV; the first unnamed column is just the row number."""
    df = pd.read_csv(INPUT_FILE, index_col=0)
    print(f'Loaded {INPUT_FILE}: {df.shape[0]} rows, {df.shape[1]} columns')
    return df


def rename_columns(df):
    """Step 2: 'Saving accounts' -> 'saving_accounts'."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print(f'Columns: {list(df.columns)}')
    return df


def remove_duplicates(df):
    """Step 3: drop identical rows."""
    n = df.duplicated().sum()
    print(f'Duplicate rows: {n}')
    return df.drop_duplicates().reset_index(drop=True)


def clean_text(df):
    """Step 4: strip spaces and lowercase all text values, then show the categories."""
    for col in CATEGORY_COLS:
        df[col] = df[col].str.strip().str.lower()
    for col in CATEGORY_COLS:
        print(f'  {col}: {sorted(df[col].dropna().unique())}')
    return df


def fill_missing(df):
    """Step 5: a missing account means the applicant has no account."""
    print('Missing values before:')
    print(df.isna().sum()[df.isna().sum() > 0].to_string())
    df[['saving_accounts', 'checking_account']] = df[[
        'saving_accounts', 'checking_account']].fillna('none')
    assert df.isna().sum().sum() == 0, 'there are still missing values'
    print('Missing values after: 0')
    return df


def fix_types(df):
    """Step 6: job codes are categories (skill levels), not quantities."""
    df['job'] = df['job'].map(JOB_LABELS)
    print(f'Job labels: {df["job"].value_counts().to_dict()}')
    return df


def validate(df):
    """Step 7: check that every value is in a sensible range."""
    checks = {
        'age between 18 and 100': df['age'].between(18, 100),
        'credit_amount > 0': df['credit_amount'] > 0,
        'duration > 0': df['duration'] > 0,
        'risk is good/bad': df['risk'].isin(['good', 'bad']),
        'job label is known': df['job'].notna(),
    }
    for name, ok in checks.items():
        print(f'  {name}: {"OK" if ok.all() else f"{(~ok).sum()} rows FAIL"}')
        if not ok.all():
            print(df[~ok])
    return df


def report_outliers(df):
    """Step 8: flag outliers with the IQR rule, but keep them (they are real applicants)."""
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = ((df[col] < low) | (df[col] > high)).sum()
        print(f'  {col}: {n} outliers (outside {low:.0f} to {high:.0f}) - kept')
    return df


def encode_target(df):
    """Step 9: risk_good = 1 if the applicant repaid, 0 if not."""
    df['risk_good'] = (df['risk'] == 'good').astype(int)
    print(f'Target: {df["risk_good"].sum()} good, {(df["risk_good"] == 0).sum()} bad '
          f'({df["risk_good"].mean():.0%} good)')
    return df


def main():
    print('Step 1: load')
    df = load_data()
    rows_before = len(df)

    print('\nStep 2: rename columns')
    df = rename_columns(df)

    print('\nStep 3: duplicates')
    df = remove_duplicates(df)

    print('\nStep 4: clean text')
    df = clean_text(df)

    print('\nStep 5: missing values')
    df = fill_missing(df)

    print('\nStep 6: fix types')
    df = fix_types(df)

    print('\nStep 7: validate')
    df = validate(df)

    print('\nStep 8: outliers')
    df = report_outliers(df)

    print('\nStep 9: target')
    df = encode_target(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f'\nStep 10: saved {OUTPUT_FILE}: {rows_before} rows in, {len(df)} rows out, '
          f'{df.shape[1]} columns')


if __name__ == '__main__':
    main()
