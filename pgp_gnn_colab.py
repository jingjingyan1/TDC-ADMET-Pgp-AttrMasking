"""
Pgp_Broccatelli benchmark - GNN (AttrMasking & ContextPred) via DeepPurpose
Run on Google Colab with A100 GPU
Copy each section into a separate Colab cell
"""

# ============================================================
# Cell 1: Install dependencies
# ============================================================
# !pip install pytdc DeepPurpose dgl dgllife

# ============================================================
# Cell 2: Imports
# ============================================================
import numpy as np
from tdc.benchmark_group import admet_group
from DeepPurpose import utils as dp_utils
from DeepPurpose import CompoundPred

# ============================================================
# Cell 3: Define training function
# ============================================================
def run_benchmark(drug_encoding, model_params, seeds=[1, 2, 3, 4, 5]):
    """
    Run Pgp_Broccatelli benchmark with a given drug encoding.
    drug_encoding: 'DGL_GIN_AttrMasking' or 'DGL_GIN_ContextPred'
    """
    group = admet_group(path='data/')
    predictions_list = []

    for seed in seeds:
        print(f"\n----- Seed {seed} ({drug_encoding}) -----")
        benchmark = group.get('Pgp_Broccatelli')
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(
            benchmark=name, split_type='default', seed=seed
        )

        # Prepare data for DeepPurpose
        train_data = dp_utils.data_process(
            X_drug=train['Drug'].values,
            y=train['Y'].values,
            drug_encoding=drug_encoding,
            random_seed=seed,
            split_method='no_split'
        )

        valid_data = dp_utils.data_process(
            X_drug=valid['Drug'].values,
            y=valid['Y'].values,
            drug_encoding=drug_encoding,
            random_seed=seed,
            split_method='no_split'
        )

        test_data = dp_utils.data_process(
            X_drug=test['Drug'].values,
            y=test['Y'].values,
            drug_encoding=drug_encoding,
            random_seed=seed,
            split_method='no_split'
        )

        # Configure model
        config = dp_utils.generate_config(
            drug_encoding=drug_encoding,
            cls_hidden_dims=[512, 128],
            train_epoch=100,
            LR=0.0005,
            batch_size=128,
            **model_params
        )

        model = CompoundPred.model_initialize(**config)
        model.train(train_data, valid_data, test_data)

        # Get predictions
        y_pred = model.predict(test_data)

        predictions = {name: y_pred}
        predictions_list.append(predictions)

        single_result = group.evaluate(predictions)
        print(f"  Result: {single_result}")

    # Final results
    final_results = group.evaluate_many(predictions_list)
    return final_results

# ============================================================
# Cell 4: Run AttrMasking
# ============================================================
print("=" * 60)
print("Running AttrMasking (pretrained GNN)")
print("=" * 60)

attr_results = run_benchmark(
    drug_encoding='DGL_GIN_AttrMasking',
    model_params={}
)

print(f"\nAttrMasking final results:")
for dataset, values in attr_results.items():
    print(f"  {dataset}: {values[0]:.3f} +/- {values[1]:.3f}")

# ============================================================
# Cell 5: Run ContextPred
# ============================================================
print("=" * 60)
print("Running ContextPred (pretrained GNN)")
print("=" * 60)

ctx_results = run_benchmark(
    drug_encoding='DGL_GIN_ContextPred',
    model_params={}
)

print(f"\nContextPred final results:")
for dataset, values in ctx_results.items():
    print(f"  {dataset}: {values[0]:.3f} +/- {values[1]:.3f}")

# ============================================================
# Cell 6: Compare all results
# ============================================================
print("\n" + "=" * 60)
print("Summary - All Models")
print("=" * 60)
print(f"  XGBoost + Morgan FP:  0.912 +/- 0.007  (your previous best)")
print(f"  AttrMasking (GNN):", end="  ")
for dataset, values in attr_results.items():
    print(f"{values[0]:.3f} +/- {values[1]:.3f}")
print(f"  ContextPred (GNN):", end="  ")
for dataset, values in ctx_results.items():
    print(f"{values[0]:.3f} +/- {values[1]:.3f}")
