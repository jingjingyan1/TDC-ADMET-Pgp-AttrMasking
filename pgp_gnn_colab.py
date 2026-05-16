"""
Pgp_Broccatelli benchmark - AttrMasking (pretrained GNN) via DeepPurpose

Setup:
    !pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
    !pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
    !pip install dgllife DeepPurpose pytdc
    !pip install git+https://github.com/bp-kelley/descriptastorus pandas-flavor
    # Then restart runtime

Run:
    !python pgp_gnn.py 2>&1 | tee results.txt
"""

import numpy as np
from tdc.benchmark_group import admet_group
from DeepPurpose import utils as dp_utils
from DeepPurpose import CompoundPred

# ---- Config ----
DRUG_ENCODING = 'DGL_GIN_AttrMasking'
DATASET = 'Pgp_Broccatelli'
SEEDS = [1, 2, 3, 4, 5]

# ---- Run benchmark ----
group = admet_group(path='data/')
predictions_list = []

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")

    benchmark = group.get(DATASET)
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    train, valid = group.get_train_valid_split(
        benchmark=name, split_type='default', seed=seed
    )

    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # Prepare data
    train_data = dp_utils.data_process(
        X_drug=train['Drug'].values,
        y=train['Y'].values,
        drug_encoding=DRUG_ENCODING,
        random_seed=seed,
        split_method='no_split'
    )
    valid_data = dp_utils.data_process(
        X_drug=valid['Drug'].values,
        y=valid['Y'].values,
        drug_encoding=DRUG_ENCODING,
        random_seed=seed,
        split_method='no_split'
    )
    test_data = dp_utils.data_process(
        X_drug=test['Drug'].values,
        y=test['Y'].values,
        drug_encoding=DRUG_ENCODING,
        random_seed=seed,
        split_method='no_split'
    )

    # Train
    config = dp_utils.generate_config(
        drug_encoding=DRUG_ENCODING,
        cls_hidden_dims=[512, 128],
        train_epoch=100,
        LR=0.0005,
        batch_size=128
    )
    model = CompoundPred.model_initialize(**config)
    model.train(train_data, valid_data, test_data)

    # Predict and evaluate
    y_pred = model.predict(test_data)
    predictions = {name: y_pred}
    predictions_list.append(predictions)

    single_result = group.evaluate(predictions)
    print(f"\nSeed {seed} result: {single_result}")

# ---- Final results ----
final_results = group.evaluate_many(predictions_list)
print(f"\n{'='*50}")
print("Final results (mean +/- std)")
print(f"{'='*50}")
for dataset, values in final_results.items():
    print(f"  {dataset}: {values[0]:.3f} +/- {values[1]:.3f}")
