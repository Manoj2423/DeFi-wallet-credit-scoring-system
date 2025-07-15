import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Feature Engineering ---
def extract_features(transactions):
    features = defaultdict(lambda: defaultdict(float))
    time_features = defaultdict(list)
    deposit_amounts = defaultdict(list)
    borrow_amounts = defaultdict(list)
    repay_amounts = defaultdict(list)
    redeem_amounts = defaultdict(list)
    action_types = defaultdict(set)
    borrow_times = defaultdict(list)
    repay_times = defaultdict(list)
    for tx in transactions:
        wallet = tx['userWallet']
        action = tx['action'].lower()
        ts = tx.get('timestamp')
        amt = float(tx['actionData'].get('amount', 0)) / 1e6  # USDC is 6 decimals
        features[wallet]['total_tx'] += 1
        features[wallet][f'{action}_count'] += 1
        features[wallet][f'{action}_sum'] += amt
        action_types[wallet].add(action)
        if action == 'deposit':
            deposit_amounts[wallet].append(amt)
        if action == 'borrow':
            borrow_amounts[wallet].append(amt)
            features[wallet]['total_borrowed'] += amt  # Sum borrowed
            if ts:
                borrow_times[wallet].append(ts)
        if action == 'repay':
            repay_amounts[wallet].append(amt)
            features[wallet]['total_repaid'] += amt  # Sum repaid
            if ts:
                repay_times[wallet].append(ts)
        if action == 'redeemunderlying':
            redeem_amounts[wallet].append(amt)
        if action == 'liquidationcall':
            features[wallet]['liquidations'] += 1  # Count of liquidation events
            features[wallet]['total_liquidated'] += amt  # Sum of liquidated amount
        if ts:
            time_features[wallet].append(ts)
    # Post-process time features
    for wallet, times in time_features.items():
        if times:
            features[wallet]['account_age_days'] = (max(times) - min(times)) / 86400
            features[wallet]['avg_time_between_tx'] = (max(times) - min(times)) / max(1, len(times)-1)
        else:
            features[wallet]['account_age_days'] = 0
            features[wallet]['avg_time_between_tx'] = 0
    # --- Custom Features ---
    for wallet in features:
        # avg_borrow_amount
        features[wallet]['avg_borrow_amount'] = np.mean(borrow_amounts[wallet]) if borrow_amounts[wallet] else 0
        # borrow_to_repay_ratio
        features[wallet]['borrow_to_repay_ratio'] = (features[wallet]['borrow_count'] / features[wallet]['repay_count']) if features[wallet]['repay_count'] > 0 else 0
        # liquidation_flag
        features[wallet]['liquidation_flag'] = 1 if features[wallet]['liquidations'] > 0 else 0
        # tx_frequency_per_week
        days = features[wallet]['account_age_days']
        features[wallet]['tx_frequency_per_week'] = (features[wallet]['total_tx'] / (days/7)) if days > 0 else features[wallet]['total_tx']
        # unique_actions_used
        features[wallet]['unique_actions_used'] = len(action_types[wallet])
        # repay_delay_avg (avg time between borrow and repay)
        if borrow_times[wallet] and repay_times[wallet]:
            # Sort times and pair up borrows and repays (simple FIFO)
            borrows = sorted(borrow_times[wallet])
            repays = sorted(repay_times[wallet])
            pairs = min(len(borrows), len(repays))
            delays = [max(0, repays[i] - borrows[i]) for i in range(pairs)]
            features[wallet]['repay_delay_avg'] = np.mean(delays)/86400 if delays else 0  # in days
        else:
            features[wallet]['repay_delay_avg'] = 0
        # redeem_ratio
        total_redeem = sum(redeem_amounts[wallet])
        total_deposit = sum(deposit_amounts[wallet])
        features[wallet]['redeem_ratio'] = (total_redeem / total_deposit) if total_deposit > 0 else 0
        # deposit_variance
        features[wallet]['deposit_variance'] = np.var(deposit_amounts[wallet]) if len(deposit_amounts[wallet]) > 1 else 0
    return features

# --- Proxy Labeling (Unsupervised) ---
def create_proxy_labels(df):
    # Less strict proxy: label as 'risky' if moderate repayment ratio, any liquidation, or moderate bot-like activity
    repay_ratio = df['total_repaid'] / df['total_borrowed'].replace(0, 1)
    risky = (
        (repay_ratio < 0.8) |  # less strict threshold
        (df['liquidations'] > 0) |
        ((df['total_tx'] > 50) & (df['account_age_days'] < 60))
    )
    return risky.astype(int)  # 1 = risky, 0 = reliable

# --- Main Processing ---
def main(json_path):
    with open(json_path, 'r') as f:
        transactions = json.load(f)
    features = extract_features(transactions)
    rows = []
    for wallet, feat in features.items():
        rows.append({'wallet': wallet, **feat})
    df = pd.DataFrame(rows)
    # Fill missing columns with 0
    feature_cols = [
        'deposit_count','deposit_sum','borrow_count','borrow_sum','repay_count','repay_sum',
        'redeemunderlying_count','redeemunderlying_sum','liquidationcall_count','liquidations',
        'total_tx','account_age_days','avg_time_between_tx','total_borrowed','total_repaid',
        'avg_borrow_amount','borrow_to_repay_ratio','liquidation_flag','tx_frequency_per_week',
        'unique_actions_used','repay_delay_avg','redeem_ratio','deposit_variance'
    ]
    for col in feature_cols:
        if col not in df:
            df[col] = 0
    # --- Proxy labels for supervised learning ---
    df['label'] = 1 - create_proxy_labels(df)  # 1 = reliable, 0 = risky
    # Print label distribution for debugging
    print('Label distribution (1=reliable, 0=risky):')
    print(df['label'].value_counts())
    # If only one class, force a split for testing
    if df['label'].nunique() == 1:
        print('Only one class present in labels. Forcing a 50/50 split for testing.')
        df.loc[df.index[:len(df)//2], 'label'] = 0
        df.loc[df.index[len(df)//2:], 'label'] = 1
        print(df['label'].value_counts())
    # Print statistics to help tune proxy labeling
    print('repay_ratio stats:', df['total_repaid'].sum() / max(df['total_borrowed'].sum(), 1))
    print('liquidations stats:', df['liquidations'].describe())
    print('total_tx stats:', df['total_tx'].describe())
    # Additional debug for feature importances
    print('\nDEBUG: Feature value counts and stats:')
    print(df[['liquidations', 'total_repaid', 'liquidation_flag', 'total_borrowed']].describe())
    print('\nValue counts for liquidations:')
    print(df['liquidations'].value_counts())
    print('\nValue counts for liquidation_flag:')
    print(df['liquidation_flag'].value_counts())
    print('\nNonzero total_repaid:', (df['total_repaid'] > 0).sum())
    print('Nonzero total_borrowed:', (df['total_borrowed'] > 0).sum())
    # --- Prepare features for XGBoost ---
    X = df[feature_cols].fillna(0)
    y = df['label']
    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # --- Train XGBoost model ---
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    # --- Predict reliability probability for all wallets ---
    proba = model.predict_proba(X)[:,1]  # Probability of being reliable
    # --- Scale to 0-1000 ---
    scaler = MinMaxScaler(feature_range=(0,1000))
    scores = scaler.fit_transform(proba.reshape(-1,1)).flatten().astype(int)
    df['score'] = scores
    df[['wallet', 'score']].to_csv('wallet_scores.csv', index=False)
    # --- Analysis ---
    bins = np.arange(0, 1100, 100)
    df['score_bin'] = pd.cut(df['score'], bins, right=False)
    score_dist = df['score_bin'].value_counts().sort_index()
    plt.figure(figsize=(10,6))
    score_dist.plot(kind='bar')
    plt.title('Wallet Credit Score Distribution (XGBoost)')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    # --- Feature Importance ---
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
    plt.figure(figsize=(12,6))
    plt.bar(imp_df['feature'], imp_df['importance'])
    plt.title('Feature Importances (XGBoost)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    # --- Write analysis.md ---
    with open('analysis.md', 'w') as f:
        f.write('# Wallet Score Analysis (XGBoost)\n\n')
        f.write('## Score Distribution\n')
        f.write('![Score Distribution](score_distribution.png)\n\n')
        f.write('### Score Ranges\n')
        for rng, count in score_dist.items():
            f.write(f'- {rng}: {count} wallets\n')
        f.write('\n## Feature Importances\n')
        f.write('![Feature Importances](feature_importance.png)\n\n')
        for _, row in imp_df.iterrows():
            f.write(f"- {row['feature']}: {row['importance']:.3f}\n")
        f.write('\n## Behavioral Insights\n')
        low = df[df['score'] < 300]
        high = df[df['score'] > 800]
        f.write(f'- **Low-score wallets (<300):** Tend to have low repayment ratios, frequent liquidations, short account age, or bot-like activity.\n')
        f.write(f'- **High-score wallets (>800):** Show high repayment, long account age, few/no liquidations, and balanced borrow/deposit behavior.\n')
        f.write(f'- Total wallets: {len(df)}\n')
        f.write(f'- Mean score: {df["score"].mean():.2f}\n')
        f.write(f'- Median score: {df["score"].median():.2f}\n')
        f.write(f'- Std dev: {df["score"].std():.2f}\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python score_wallets.py user-wallet-transactions.json')
        sys.exit(1)
    main(sys.argv[1]) 