# DeFi Wallet Credit Scoring

## Introduction
This project assigns a credit score (0-1000) to each wallet interacting with the Aave V2 protocol, based solely on historical transaction behavior. The score reflects the reliability and risk profile of each wallet, with higher scores indicating responsible usage and lower scores indicating risky or exploitative behavior.

## Method Chosen
We use **Gradient Boosted Trees (XGBoost)**, a state-of-the-art machine learning algorithm for tabular data, to model wallet reliability. Since real-world “good/bad” labels are not available, we generate **proxy labels** based on transaction patterns (e.g., repayment ratio, liquidations, bot-like activity). The model is trained to predict the probability of a wallet being “reliable,” which is then scaled to a 0–1000 credit score.

## Complete Architecture
1. **Input:**
   - Raw transaction-level JSON data from Aave V2, with each record representing a wallet action (deposit, borrow, repay, redeemunderlying, liquidationcall).
2. **Feature Engineering:**
   - Extract features per wallet, such as transaction counts, sums, ratios, timing, action diversity, and behavioral flags (see table below).
3. **Proxy Labeling:**
   - Assign a “reliable” or “risky” label to each wallet using rules based on repayment, liquidations, and activity patterns.
4. **Model Training:**
   - Train an XGBoost classifier to predict reliability from features.
5. **Scoring:**
   - Use the model’s output probability, scaled to 0–1000, as the wallet’s credit score.
6. **Output:**
   - Generate `wallet_scores.csv` (wallet and score), `analysis.md` (score distribution, feature importances, behavioral insights), and graphs (`score_distribution.png`, `feature_importance.png`).

**Architecture Diagram:**

```
Raw JSON Data
      |
      v
Feature Engineering (Python)
      |
      v
Proxy Labeling (Rules)
      |
      v
XGBoost Model Training
      |
      v
Score Scaling (0-1000)
      |
      v
Outputs: CSV, Markdown, Graphs
```

## Processing Flow
1. **Load transaction data** from JSON file.
2. **Aggregate features** for each wallet (see feature table below).
3. **Generate proxy labels** for supervised learning.
4. **Split data** into training and test sets.
5. **Train XGBoost model** to predict reliability.
6. **Score each wallet** using the model’s output probability, scaled to 0–1000.
7. **Save results** to CSV and generate analysis/graphs.

## Feature Engineering
The following features are extracted and used in the machine learning model:

| Feature Name            | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `deposit_count`         | Number of deposit transactions                                |
| `deposit_sum`           | Cumulative deposited amount                                   |
| `borrow_count`          | Number of borrow actions                                      |
| `avg_borrow_amount`     | Average borrowed value per transaction                        |
| `repay_count`           | Count of repayment transactions                               |
| `borrow_to_repay_ratio` | Ratio of borrows to repayments                                |
| `liquidation_flag`      | Binary flag if wallet was liquidated at least once            |
| `liquidations`          | Number of liquidations                                        |
| `total_liquidated`      | Cumulative amount liquidated                                  |
| `account_age_days`      | Number of days between first and last tx                      |
| `tx_frequency_per_week` | Normalized frequency of interactions                          |
| `unique_actions_used`   | Count of unique transaction types used                        |
| `repay_delay_avg`       | Avg time (days) between borrow and repay                      |
| `redeem_ratio`          | Total redeem / total deposit                                  |
| `deposit_variance`      | Variance of deposit amounts (bots often repeat fixed amounts) |
| `total_tx`              | Total number of transactions                                  |
| `avg_time_between_tx`   | Average time between transactions (seconds)                   |
| `total_borrowed`        | Total borrowed amount                                         |
| `total_repaid`          | Total repaid amount                                           |
| `repay_sum`             | Cumulative repaid amount                                      |
| `redeemunderlying_sum`  | Cumulative redeemed amount                                    |
| `redeemunderlying_count`| Number of redeem actions                                      |
| `liquidationcall_count` | Number of liquidation call actions                            |

These features are used as input to the XGBoost model and are included in the feature importance analysis.

## Installation

1. **Download requirements.txt** (if not already present in your directory).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run/Extend
1. **Run the script:**
   ```bash
   python score_wallets.py user-wallet-transactions.json
   ```
2. **Outputs:**
   - `wallet_scores.csv`, `analysis.md`, `score_distribution.png`, `feature_importance.png`

To extend, modify the feature engineering or model logic in `score_wallets.py`.

## Transparency & Extensibility
- The scoring method and features are fully documented.
- The code is modular and commented for easy extension.
- Feature importances are provided for interpretability. 