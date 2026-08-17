import json
with open("/leonardo_scratch/fast/EUHPC_D36_033/ogas_states_runs/ks800_v3_pilot/campaign_analysis_rmse_mean.json") as f:
    d = json.load(f)

for row in d['rows']:
    if row['arm'] == 'uniform_baseline' and 'hard_val/val_tube_' in row['metric']:
        print(f"RMSE {row['metric']}: {row['mean']:.4f}")
