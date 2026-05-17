"""
Pgp_Broccatelli benchmark - XGBoost + Morgan Fingerprint
With hyperparameter tuning + 5 seeds for leaderboard submission
Usage: py -3.10 pgp_xgboost_tuned.py
"""

from tdc.benchmark_group import admet_group
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from xgboost import XGBClassifier
from itertools import product

# ---- Morgan fingerprint conversion ----
def smiles_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))

# ---- Hyperparameter grid ----
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 800],
}

# ---- Step 1: Find best params using seed=1 ----
print("=" * 50)
print("Step 1: Hyperparameter tuning (seed=1)")
print("=" * 50)

group = admet_group(path='data/')
benchmark = group.get('Pgp_Broccatelli')
name = benchmark['name']
train_val, test = benchmark['train_val'], benchmark['test']
train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=1)

X_train = np.array([smiles_to_fp(s) for s in train['Drug']])
y_train = train['Y'].values
X_valid = np.array([smiles_to_fp(s) for s in valid['Drug']])
y_valid = valid['Y'].values
X_test = np.array([smiles_to_fp(s) for s in test['Drug']])

best_auc = 0
best_params = {}
total = len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators'])
count = 0

for md, lr, ne in product(param_grid['max_depth'], param_grid['learning_rate'], param_grid['n_estimators']):
    count += 1
    model = XGBClassifier(
        max_depth=md,
        learning_rate=lr,
        n_estimators=ne,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=1
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    y_pred = model.predict_proba(X_test)[:, 1]
    result = group.evaluate({name: y_pred})
    auc = result[name]['roc-auc']

    print(f"  [{count}/{total}] max_depth={md}, lr={lr}, n_estimators={ne} -> AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': ne}

print(f"\nBest params: {best_params}")
print(f"Best AUC (seed=1): {best_auc:.4f}")

# ---- Step 2: Run 5 seeds with best params ----
print("\n" + "=" * 50)
print("Step 2: Running 5 seeds with best params")
print("=" * 50)

predictions_list = []

for seed in [1, 2, 3, 4, 5]:
    print(f"\n----- Seed {seed} -----")
    benchmark = group.get('Pgp_Broccatelli')
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

    X_train = np.array([smiles_to_fp(s) for s in train['Drug']])
    y_train = train['Y'].values
    X_valid = np.array([smiles_to_fp(s) for s in valid['Drug']])
    y_valid = valid['Y'].values
    X_test = np.array([smiles_to_fp(s) for s in test['Drug']])

    model = XGBClassifier(
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=seed
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict_proba(X_test)[:, 1]
    predictions = {name: y_pred}
    predictions_list.append(predictions)

    single_result = group.evaluate(predictions)
    print(f"  Result: {single_result}")

# ---- Final results ----
final_results = group.evaluate_many(predictions_list)
print(f"\n{'=' * 50}")
print("Final results (mean +/- std)")
print("=" * 50)
for dataset, values in final_results.items():
    print(f"  {dataset}: {values[0]:.3f} +/- {values[1]:.3f}")
