# Generative-designs-for-uavs

## Live dashboard

The Matplotlib live visualizer has been replaced with a Streamlit dashboard.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run the optimizer in one terminal:

```bash
python main.py
```

Run the dashboard in another terminal:

```bash
streamlit run dashboard.py
```

`live_visualizer.py` is kept as a compatibility launcher and now opens the Streamlit dashboard instead of a Matplotlib window.

If `scikit-learn` is not installed, the optimizer will still run, but it will disable the surrogate model and skip ML predictions until you install it.

## Dataset generation

Use the physics-based collector to build training data from XFOIL:

```bash
python dataset_collector.py --samples 1000 --batch-size 100 --workers 4
```

It writes:

- `uav_dataset.csv`
- `uav_dataset.jsonl`
- `uav_dataset_summary.json`
- `uav_dataset_cache.json`

The collector uses a 70/30 random-plus-GA hybrid sampler, batches evaluations, and skips invalid XFOIL results automatically.

The summary JSON now includes dataset-level metrics such as `total_samples`, `valid_samples`, `failed_runs`, `avg_Cl`, `avg_Cd`, `avg_LD`, `aspect_ratio`, and `failure_reason_counts`.

Before training ML, validate the dataset distributions:

```bash
python dataset_validator.py --csv uav_dataset.csv --output-dir dataset_validation
```

This generates histograms for `Cl`, `Cd`, `L/D`, `aspect_ratio`, `wing_span`, and `wing_area`, plus a validation report JSON with bias checks and red-flag warnings.

## Statistical experiment

Run the multi-seed comparison for GA, GA + ML, and GA + ML + RL:

```bash
python statistical_experiment.py
```

It writes aggregated outputs to `.statistical_experiment/`, including JSON, CSV, a summary text file, and a comparison plot with mean and standard deviation.
