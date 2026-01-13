from ultralytics import YOLO
from pathlib import Path
import torch
import shutil
import glob
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# workspace root
project_root = Path(__file__).resolve().parents[2]
results_root = Path(__file__).resolve().parent / "runs" / "train"
trained_models_dir = project_root / "trained_models"
ensure_dir(results_root)
ensure_dir(trained_models_dir)

#  device
# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


model = YOLO("yolov8n.pt")

run_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

# train
data_path = project_root / "data" / "data.yaml"
if not data_path.exists():
    raise FileNotFoundError(f"Dataset file not found: {data_path}\nPlease ensure data.yaml exists at this path or update the script.")

model.train(
    data=str(data_path),
    epochs=40,
    imgsz=640,
    batch=8,
    patience=10,
    device=device,
    project=str(results_root),
    name=run_name,
    # rate scheduler 
    lr0=0.01,           # initial lr
    lrf=0.01,           # final lr 
    warmup_epochs=3,   
    warmup_momentum=0.8,  
    warmup_bias_lr=0.1, 
)

# locate latest run folder
run_dir = None
candidate_dirs = sorted(results_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
for d in candidate_dirs:
    if d.name == run_name or d.is_dir():
        run_dir = d
        break

if run_dir is None:
    raise RuntimeError("Could not find training run directory under runs/train")


results_dir = run_dir

# csv
metrics_csv = None
for pattern in ("metrics.csv", "results.csv", "metrics*.csv", "*.csv"):
    found = list(results_dir.rglob(pattern))
    if found:
        # pick the most recent csv
        metrics_csv = sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        break

training_history_csv = results_dir / "training_history.csv"
if metrics_csv is not None:
    try:
        df = pd.read_csv(metrics_csv)
        df.to_csv(training_history_csv, index=False)
    except Exception:
        # if can't read, fallback to copying raw file
        shutil.copy(str(metrics_csv), str(training_history_csv))

# generate plots if we have a CSV
plots_dir = results_dir / "plots"
ensure_dir(plots_dir)
if training_history_csv.exists():
    try:
        df = pd.read_csv(training_history_csv)
        # try common columns
        cols = df.columns.str.lower()
        # loss plot
        loss_cols = [c for c in df.columns if "loss" in c.lower()]
        if loss_cols:
            plt.figure()
            for c in loss_cols:
                plt.plot(df[c], label=c)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "loss.png")
            plt.close()

        # metrics: precision, recall, map
        metric_keys = {"precision": None, "recall": None, "map": None}
        for c in df.columns:
            cl = c.lower()
            if "precision" in cl and metric_keys['precision'] is None:
                metric_keys['precision'] = c
            if "recall" in cl and metric_keys['recall'] is None:
                metric_keys['recall'] = c
            if ("map" in cl or "mAP" in c) and metric_keys['map'] is None:
                metric_keys['map'] = c

        plt.figure()
        plotted = False
        for name, col in metric_keys.items():
            if col and col in df.columns:
                plt.plot(df[col], label=col)
                plotted = True
        if plotted:
            plt.xlabel('epoch')
            plt.ylabel('metric')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "metrics.png")
            plt.close()
    except Exception:
        pass

# Save test metrics: try to load best.pt and run validation
test_metrics = {}
best_pt = None
weights_dir = results_dir / "weights"
if weights_dir.exists():
    b = weights_dir / "best.pt"
    l = weights_dir / "last.pt"
    if b.exists():
        best_pt = b
    elif l.exists():
        best_pt = l

if best_pt is not None:
    try:
        best_model = YOLO(str(best_pt))
        val_res = best_model.val(data=str(data_path))
        # Try to extract typical metrics from val_res
        if hasattr(val_res, 'metrics') and isinstance(val_res.metrics, dict):
            test_metrics.update(val_res.metrics)
        else:
            # val_res may have a results list or dict
            try:
                # some versions return a Results object with .stats or .box
                if hasattr(val_res, 'box'):
                    test_metrics['boxes'] = True
            except Exception:
                pass
    except Exception:
        pass

# If test metrics empty, try to grab final row from training CSV
if not test_metrics and training_history_csv.exists():
    try:
        df = pd.read_csv(training_history_csv)
        last = df.iloc[-1].to_dict()
        # map some names
        precision = None
        recall = None
        mAP = None
        for k, v in last.items():
            kl = k.lower()
            if 'precision' in kl and precision is None:
                precision = float(v) if pd.notna(v) else None
            if 'recall' in kl and recall is None:
                recall = float(v) if pd.notna(v) else None
            if ('map' in kl) and mAP is None:
                try:
                    mAP = float(v)
                except Exception:
                    mAP = None

        # compute approximate f1
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = None

        test_metrics.update({
            "test_precision_macro": precision,
            "test_recall_macro": recall,
            "test_f1_macro": f1,
            "test_map": mAP,
        })
    except Exception:
        pass

# write test metrics json
test_metrics_path = results_dir / "test_metrics.json"
with open(test_metrics_path, 'w') as f:
    json.dump(test_metrics, f, indent=2)

# Copy and number models into trained_models
def next_model_index(dest_dir: Path):
    existing = sorted(dest_dir.glob("model_*.pt"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        name = p.stem
        try:
            n = int(name.split('_')[1])
            nums.append(n)
        except Exception:
            continue
    return max(nums) + 1 if nums else 1

idx = next_model_index(trained_models_dir)
if weights_dir.exists():
    for w in (weights_dir / "best.pt", weights_dir / "last.pt"):
        if w.exists():
            suffix = w.stem  # best or last
            dest = trained_models_dir / f"model_{idx:03d}_{suffix}.pt"
            shutil.copy(str(w), str(dest))

print(f"Training complete. Results saved in: {results_dir}")
print(f"Training history: {training_history_csv}")
print(f"Test metrics: {test_metrics_path}")
print(f"Saved models to: {trained_models_dir}")
