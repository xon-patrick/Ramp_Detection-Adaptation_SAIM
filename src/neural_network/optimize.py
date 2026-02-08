"""
Optimization script for YOLOv8 ramp detection.
Runs multiple experiments, compares metrics, generates plots,
exports best model to ONNX, and produces analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


@dataclass
class ExperimentResult:
	exp_id: str
	name: str
	changes: str
	accuracy: float | None
	f1_macro: float | None
	precision_macro: float | None
	recall_macro: float | None
	map50: float | None
	map: float | None
	train_time_sec: float | None
	run_name: str
	status: str
	notes: str


def _load_yaml(path: Path) -> dict:
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		yaml.safe_dump(data, f, sort_keys=False)


def _resolve_path(project_root: Path, value: str | Path) -> Path:
	value_path = Path(str(value))
	if value_path.is_absolute():
		return value_path
	return (project_root / value_path).resolve()


def _relative_path(project_root: Path, path: Path) -> str:
	try:
		return str(path.relative_to(project_root))
	except ValueError:
		return str(path)


def _find_test_images_dir(project_root: Path, data_yaml_path: Path) -> Path:
	data_cfg = _load_yaml(data_yaml_path)
	yaml_dir = data_yaml_path.parent

	base_path = None
	if "path" in data_cfg:
		base_path = _resolve_path(project_root, data_cfg["path"])

	test_value = data_cfg.get("test")
	tried: list[Path] = []

	if test_value:
		test_path = _resolve_path(project_root, test_value)
		if base_path is not None:
			test_path = _resolve_path(base_path, test_value)
		tried.append(test_path)
		if test_path.exists():
			return test_path / "images" if (test_path / "images").exists() else test_path

	candidates = [
		project_root / "test" / "images",
		project_root / "data" / "test" / "images",
		yaml_dir / "test" / "images",
	]
	for candidate in candidates:
		tried.append(candidate)
		if candidate.exists():
			return candidate

	tried_text = "\n".join(f"- {path}" for path in tried)
	raise FileNotFoundError(
		"Could not locate test images directory. Tried:\n" + tried_text
	)


def _get_class_names(data_yaml_path: Path) -> list[str]:
	data_cfg = _load_yaml(data_yaml_path)
	names = data_cfg.get("names")
	if isinstance(names, dict):
		return [names[k] for k in sorted(names)]
	if isinstance(names, list):
		return names
	return ["class_0", "class_1", "class_2"]


def _compute_metrics_from_val(results, class_names: list[str]) -> dict:
	precision_list = results.box.p
	recall_list = results.box.r

	f1_per_class = 2 * (precision_list * recall_list) / (precision_list + recall_list + 1e-6)

	ap50_list = getattr(results.box, "ap50", None)
	ap_list = getattr(results.box, "ap", None)

	per_class = {}
	for idx, name in enumerate(class_names):
		per_class[name] = {
			"precision": float(precision_list[idx]),
			"recall": float(recall_list[idx]),
			"mAP50": float(ap50_list[idx]) if ap50_list is not None else None,
			"mAP50-95": float(ap_list[idx]) if ap_list is not None else None,
			"f1": float(f1_per_class[idx]),
		}

	metrics = {
		"accuracy": float(results.box.map50),
		"f1_macro": float(f1_per_class.mean()),
		"precision_macro": float(precision_list.mean()),
		"recall_macro": float(recall_list.mean()),
		"map50": float(results.box.map50),
		"map": float(results.box.map),
		"per_class": per_class,
	}

	return metrics


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _save_json(path: Path, data: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)


def _load_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
	if not label_path.exists():
		return []
	labels = []
	with label_path.open("r", encoding="utf-8") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) != 5:
				continue
			cls_id, x, y, w, h = parts
			labels.append((int(cls_id), float(x), float(y), float(w), float(h)))
	return labels


def _xywhn_to_xyxy(box: tuple[float, float, float, float], img_w: int, img_h: int) -> np.ndarray:
	x, y, w, h = box
	x1 = (x - w / 2) * img_w
	y1 = (y - h / 2) * img_h
	x2 = (x + w / 2) * img_w
	y2 = (y + h / 2) * img_h
	return np.array([x1, y1, x2, y2], dtype=np.float32)


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
	x1 = max(box_a[0], box_b[0])
	y1 = max(box_a[1], box_b[1])
	x2 = min(box_a[2], box_b[2])
	y2 = min(box_a[3], box_b[3])
	inter_w = max(0.0, x2 - x1)
	inter_h = max(0.0, y2 - y1)
	inter = inter_w * inter_h
	area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
	area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
	union = area_a + area_b - inter
	return inter / union if union > 0 else 0.0


def _analyze_image_quality(image: np.ndarray) -> dict:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	brightness = float(gray.mean())
	contrast = float(gray.std())
	lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
	return {
		"brightness": brightness,
		"contrast": contrast,
		"blur": lap_var,
	}


def _infer_causes(quality: dict, box_area_ratio: float, near_edge: bool, cls_name: str) -> list[str]:
	causes = []
	if box_area_ratio < 0.02:
		causes.append("obiect mic in cadru")
	if near_edge:
		causes.append("obiect aproape de margine")
	if quality["contrast"] < 25:
		causes.append("contrast redus")
	if quality["brightness"] < 60 or quality["brightness"] > 200:
		causes.append("iluminare extrema")
	if quality["blur"] < 60:
		causes.append("blur/motion blur")
	if cls_name == "ramps-railing":
		causes.append("dezechilibru de clasa")
	return causes


def _build_confusion_and_misclassified(
	model: YOLO,
	test_images_dir: Path,
	labels_dir: Path,
	class_names: list[str],
	imgsz: int,
	device: str,
	conf: float,
	iou_threshold: float,
	max_examples: int = 5,
) -> tuple[np.ndarray, list[dict]]:
	image_paths = sorted(
		[
			*test_images_dir.glob("*.jpg"),
			*test_images_dir.glob("*.jpeg"),
			*test_images_dir.glob("*.png"),
		]
	)

	n_classes = len(class_names)
	bg_index = n_classes
	matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
	misclassified: list[dict] = []

	for image_path in image_paths:
		image = cv2.imread(str(image_path))
		if image is None:
			continue
		img_h, img_w = image.shape[:2]

		label_path = labels_dir / (image_path.stem + ".txt")
		gt_labels = _load_yolo_labels(label_path)

		pred_results = model.predict(
			source=str(image_path),
			imgsz=imgsz,
			device=device,
			conf=conf,
			iou=iou_threshold,
			verbose=False,
		)

		preds = []
		if pred_results:
			boxes = pred_results[0].boxes
			for box in boxes:
				xyxy = box.xyxy.cpu().numpy()[0]
				cls_id = int(box.cls.cpu().numpy()[0])
				conf_score = float(box.conf.cpu().numpy()[0])
				preds.append((cls_id, conf_score, xyxy))

		matched_pred = set()
		for gt_idx, (gt_cls, x, y, w, h) in enumerate(gt_labels):
			gt_box = _xywhn_to_xyxy((x, y, w, h), img_w, img_h)
			best_iou = 0.0
			best_pred = None
			best_pred_idx = None
			for pred_idx, (pred_cls, pred_conf, pred_box) in enumerate(preds):
				if pred_idx in matched_pred:
					continue
				iou = _box_iou(gt_box, pred_box)
				if iou > best_iou:
					best_iou = iou
					best_pred = (pred_cls, pred_conf, pred_box)
					best_pred_idx = pred_idx

			if best_pred is not None and best_iou >= iou_threshold:
				matched_pred.add(best_pred_idx)
				pred_cls, pred_conf, pred_box = best_pred
				matrix[gt_cls, pred_cls] += 1

				if pred_cls != gt_cls:
					quality = _analyze_image_quality(image)
					box_area_ratio = ((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])) / (img_w * img_h)
					near_edge = (
						gt_box[0] < 0.05 * img_w
						or gt_box[1] < 0.05 * img_h
						or gt_box[2] > 0.95 * img_w
						or gt_box[3] > 0.95 * img_h
					)
					causes = _infer_causes(quality, box_area_ratio, near_edge, class_names[gt_cls])
					misclassified.append(
						{
							"image": str(image_path),
							"gt_class": class_names[gt_cls],
							"pred_class": class_names[pred_cls],
							"iou": float(best_iou),
							"conf": float(pred_conf),
							"causes": causes,
						}
					)
			else:
				matrix[gt_cls, bg_index] += 1
				quality = _analyze_image_quality(image)
				box_area_ratio = ((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])) / (img_w * img_h)
				near_edge = (
					gt_box[0] < 0.05 * img_w
					or gt_box[1] < 0.05 * img_h
					or gt_box[2] > 0.95 * img_w
					or gt_box[3] > 0.95 * img_h
				)
				causes = _infer_causes(quality, box_area_ratio, near_edge, class_names[gt_cls])
				misclassified.append(
					{
						"image": str(image_path),
						"gt_class": class_names[gt_cls],
						"pred_class": "none",
						"iou": 0.0,
						"conf": 0.0,
						"causes": causes,
					}
				)

		for pred_idx, (pred_cls, pred_conf, pred_box) in enumerate(preds):
			if pred_idx in matched_pred:
				continue
			matrix[bg_index, pred_cls] += 1

	misclassified_sorted = sorted(misclassified, key=lambda x: (-x["conf"], -x["iou"]))
	return matrix, misclassified_sorted[:max_examples]


def _plot_confusion_matrix(
	matrix: np.ndarray, class_names: list[str], output_path: Path
) -> None:
	labels = class_names + ["background"]
	fig, ax = plt.subplots(figsize=(10, 8))
	im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
	ax.figure.colorbar(im, ax=ax)
	ax.set(
		xticks=np.arange(len(labels)),
		yticks=np.arange(len(labels)),
		xticklabels=labels,
		yticklabels=labels,
		ylabel="True label",
		xlabel="Predicted label",
		title="Confusion Matrix (Detection)",
	)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	thresh = matrix.max() / 2 if matrix.max() > 0 else 1
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			ax.text(
				j,
				i,
				format(matrix[i, j], "d"),
				ha="center",
				va="center",
				color="white" if matrix[i, j] > thresh else "black",
			)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def _plot_metric_comparison(
	experiments: list[ExperimentResult],
	metric_name: str,
	output_path: Path,
	title: str,
) -> None:
	labels = [exp.name for exp in experiments]
	values = [getattr(exp, metric_name) or 0.0 for exp in experiments]

	fig, ax = plt.subplots(figsize=(10, 6))
	ax.bar(labels, values, color="#4C78A8")
	ax.set_ylabel(metric_name)
	ax.set_title(title)
	ax.set_ylim(0, max(values) * 1.2 if values else 1)
	plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def _plot_learning_curves(results_csv: Path, output_path: Path, title: str) -> None:
	if not results_csv.exists():
		return

	data = np.genfromtxt(results_csv, delimiter=",", dtype=str, skip_header=0)
	headers = data[0]
	values = data[1:].astype(float)

	train_loss_idx = None
	val_loss_idx = None
	for i, header in enumerate(headers):
		header_clean = header.strip().lower()
		if "train/box_loss" == header_clean:
			train_loss_idx = i
		if "val/box_loss" == header_clean:
			val_loss_idx = i

	if train_loss_idx is None or val_loss_idx is None:
		return

	epochs = np.arange(1, len(values) + 1)
	train_loss = values[:, train_loss_idx]
	val_loss = values[:, val_loss_idx]

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(epochs, train_loss, label="Train box_loss", linewidth=2)
	ax.plot(epochs, val_loss, label="Val box_loss", linewidth=2)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def _plot_metrics_evolution(
	baseline_metrics_path: Path,
	final_metrics: dict,
	output_path: Path,
) -> None:
	if not baseline_metrics_path.exists():
		return

	with baseline_metrics_path.open("r", encoding="utf-8") as f:
		baseline = json.load(f)

	baseline_acc = float(baseline.get("test_accuracy", 0))
	baseline_f1 = float(baseline.get("test_f1_macro", 0))
	final_acc = float(final_metrics.get("accuracy", 0))
	final_f1 = float(final_metrics.get("f1_macro", 0))

	labels = ["Accuracy", "F1 macro"]
	baseline_vals = [baseline_acc, baseline_f1]
	final_vals = [final_acc, final_f1]

	x = np.arange(len(labels))
	width = 0.35

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.bar(x - width / 2, baseline_vals, width, label="Etapa 5")
	ax.bar(x + width / 2, final_vals, width, label="Optimizat")
	ax.set_ylabel("Metric")
	ax.set_title("Evolutie metrici (Etapa 5 vs Optimizat)")
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.legend()
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def _save_misclassified_report(
	project_root: Path, misclassified: list[dict], output_path: Path
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		f.write("# Analiza 5 exemple gresite\n\n")
		for idx, item in enumerate(misclassified, start=1):
			causes = ", ".join(item["causes"]) if item["causes"] else "neidentificat"
			f.write(f"## Exemplu {idx}\n")
			f.write(f"- Imagine: {_relative_path(project_root, Path(item['image']))}\n")
			f.write(f"- Clasa reala: {item['gt_class']}\n")
			f.write(f"- Predictie: {item['pred_class']}\n")
			f.write(f"- IoU: {item['iou']:.3f}\n")
			f.write(f"- Confidenta: {item['conf']:.3f}\n")
			f.write(f"- Cauze probabile: {causes}\n\n")


def _get_next_optimized_version(models_dir: Path) -> int:
	existing = list(models_dir.glob("optimized_model_v*.onnx"))
	versions = []
	for model in existing:
		try:
			versions.append(int(model.stem.split("v")[-1]))
		except ValueError:
			continue
	return max(versions) + 1 if versions else 1


def _export_best_onnx(best_model_path: Path, output_path: Path, imgsz: int) -> Path:
	model = YOLO(str(best_model_path))
	export_path = model.export(format="onnx", imgsz=imgsz, batch=1, simplify=True)
	export_path = Path(export_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_bytes(export_path.read_bytes())
	return output_path


def _train_experiment(
	project_root: Path,
	data_yaml: Path,
	exp_id: str,
	exp_name: str,
	base_cfg: dict,
	overrides: dict,
	skip_train: bool,
) -> tuple[Path | None, Path | None, float | None]:
	run_name = f"opt_{exp_id}_{exp_name}"
	run_dir = project_root / "runs" / "optimization" / run_name
	results_csv = run_dir / "results.csv"
	best_model_path = run_dir / "weights" / "best.pt"

	if skip_train and best_model_path.exists():
		return best_model_path, results_csv, None

	model = YOLO(f"{base_cfg['model_architecture']}.pt")

	train_args = {
		"data": str(data_yaml),
		"epochs": int(overrides.get("epochs", base_cfg["epochs"])),
		"batch": int(overrides.get("batch", base_cfg["batch"])),
		"imgsz": int(overrides.get("imgsz", base_cfg["imgsz"])),
		"device": str(overrides.get("device", base_cfg["device"])),
		"patience": int(overrides.get("patience", base_cfg["patience"])),
		"lr0": float(overrides.get("lr0", base_cfg["lr0"])),
		"lrf": float(overrides.get("lrf", base_cfg["lrf"])),
		"warmup_epochs": int(overrides.get("warmup_epochs", base_cfg["warmup_epochs"])),
		"save": True,
		"save_period": int(overrides.get("save_period", base_cfg["save_period"])),
		"project": str(project_root / "runs"),
		"name": f"optimization/{run_name}",
		"exist_ok": True,
		"seed": int(overrides.get("seed", base_cfg["seed"])),
		"verbose": True,
	}

	aug = base_cfg.get("augmentation", {}).copy()
	aug.update(overrides.get("augmentation", {}))
	train_args.update(aug)

	start = time.perf_counter()
	try:
		model.train(**train_args)
		end = time.perf_counter()
	except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
		if "out of memory" in str(e).lower():
			print(f"[WARNING] OOM during training: {e}")
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			return None, None, None
		raise

	if best_model_path.exists():
		return best_model_path, results_csv, end - start
	return None, results_csv, end - start


def _evaluate_model(
	model_path: Path,
	data_yaml: Path,
	imgsz: int,
	batch: int,
	device: str,
	class_names: list[str],
) -> dict:
	model = YOLO(str(model_path))
	results = model.val(
		data=str(data_yaml),
		split="test",
		imgsz=imgsz,
		batch=batch,
		device=device,
		verbose=False,
	)
	return _compute_metrics_from_val(results, class_names)


def main() -> int:
	parser = argparse.ArgumentParser(description="Run YOLOv8 optimization experiments.")
	parser.add_argument("--project-root", default="", help="Project root.")
	parser.add_argument("--config", default="config/optimized_config.yaml", help="Config path.")
	parser.add_argument("--data", default="data/data.yaml", help="Path to data.yaml.")
	parser.add_argument("--device", default="0", help="Device id.")
	parser.add_argument("--skip-train", action="store_true", help="Skip training if weights exist.")
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
	parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching.")
	args = parser.parse_args()

	if args.project_root:
		project_root = Path(args.project_root).resolve()
	else:
		project_root = Path(__file__).resolve().parents[2]
	config_path = _resolve_path(project_root, args.config)
	data_yaml = _resolve_path(project_root, args.data)

	cfg = _load_yaml(config_path)
	base_cfg = cfg.get("base", {})
	experiments_cfg = cfg.get("experiments", [])

	if not experiments_cfg:
		raise RuntimeError("No experiments defined in optimized_config.yaml")

	class_names = _get_class_names(data_yaml)

	results: list[ExperimentResult] = []
	best_result = None
	best_model_path = None
	best_results_csv = None
	best_metrics = None

	for exp in experiments_cfg:
		exp_id = str(exp.get("id", "exp"))
		exp_name = str(exp.get("name", exp_id)).replace(" ", "_")
		exp_changes = str(exp.get("changes", ""))
		overrides = exp.get("overrides", {})
		overrides = overrides.copy()
		overrides["device"] = args.device

		model_path, results_csv, train_time = _train_experiment(
			project_root=project_root,
			data_yaml=data_yaml,
			exp_id=exp_id,
			exp_name=exp_name,
			base_cfg=base_cfg,
			overrides=overrides,
			skip_train=args.skip_train,
		)

		if model_path is None:
			results.append(
				ExperimentResult(
					exp_id=exp_id,
					name=exp_name,
					changes=exp_changes,
					accuracy=None,
					f1_macro=None,
					precision_macro=None,
					recall_macro=None,
					map50=None,
					map=None,
					train_time_sec=train_time,
					run_name=exp_name,
					status="failed",
					notes="Model weights missing",
				)
			)
			continue

		metrics = _evaluate_model(
			model_path=model_path,
			data_yaml=data_yaml,
			imgsz=int(overrides.get("imgsz", base_cfg["imgsz"])),
			batch=int(overrides.get("batch", base_cfg["batch"])),
			device=str(args.device),
			class_names=class_names,
		)

		result = ExperimentResult(
			exp_id=exp_id,
			name=exp_name,
			changes=exp_changes,
			accuracy=metrics["accuracy"],
			f1_macro=metrics["f1_macro"],
			precision_macro=metrics["precision_macro"],
			recall_macro=metrics["recall_macro"],
			map50=metrics["map50"],
			map=metrics["map"],
			train_time_sec=train_time,
			run_name=exp_name,
			status="ok",
			notes=str(exp.get("notes", "")),
		)
		results.append(result)

		if best_result is None:
			best_result = result
			best_model_path = model_path
			best_results_csv = results_csv
			best_metrics = metrics
		else:
			if (result.f1_macro or 0) > (best_result.f1_macro or 0):
				best_result = result
				best_model_path = model_path
				best_results_csv = results_csv
				best_metrics = metrics

		# Cleanup GPU memory after each experiment
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	if best_result is None or best_model_path is None or best_metrics is None:
		raise RuntimeError("No successful experiments to finalize.")

	# Save experiments table
	experiments_csv_path = project_root / "results" / "optimizations_experiments.csv"
	rows = []
	for exp in results:
		rows.append(
			{
				"exp_id": exp.exp_id,
				"name": exp.name,
				"changes": exp.changes,
				"accuracy": exp.accuracy,
				"f1_macro": exp.f1_macro,
				"precision_macro": exp.precision_macro,
				"recall_macro": exp.recall_macro,
				"train_time_sec": exp.train_time_sec,
				"status": exp.status,
				"notes": exp.notes,
			}
		)

	_save_csv(
		experiments_csv_path,
		rows,
		[
			"exp_id",
			"name",
			"changes",
			"accuracy",
			"f1_macro",
			"precision_macro",
			"recall_macro",
			"train_time_sec",
			"status",
			"notes",
		],
	)

	# Export best model to ONNX
	models_dir = project_root / "models"
	version = _get_next_optimized_version(models_dir)
	onnx_path = models_dir / f"optimized_model_v{version}.onnx"
	onnx_path = _export_best_onnx(best_model_path, onnx_path, imgsz=base_cfg["imgsz"])

	# Save final metrics
	final_metrics = {
		"accuracy": best_metrics["accuracy"],
		"f1_macro": best_metrics["f1_macro"],
		"precision_macro": best_metrics["precision_macro"],
		"recall_macro": best_metrics["recall_macro"],
		"map50": best_metrics["map50"],
		"map": best_metrics["map"],
		"per_class": best_metrics["per_class"],
		"best_experiment": best_result.name,
		"onnx_model": _relative_path(project_root, onnx_path),
	}
	final_metrics_path = project_root / "results" / "final_metrics.json"
	_save_json(final_metrics_path, final_metrics)

	# Copy training history
	if best_results_csv is not None and best_results_csv.exists():
		history_path = project_root / "results" / "training_history.csv"
		history_path.write_bytes(best_results_csv.read_bytes())

		loss_curve_path = project_root / "docs" / "loss_curve.png"
		_plot_learning_curves(best_results_csv, loss_curve_path, "Loss curve (optimized)")

		learning_curve_path = project_root / "docs" / "results" / "learning_curves_final.png"
		_plot_learning_curves(best_results_csv, learning_curve_path, "Learning curves (optimized)")

	# Plots for experiments
	_plot_metric_comparison(
		results,
		"accuracy",
		project_root / "docs" / "optimization" / "accuracy_comparison.png",
		"Accuracy comparison",
	)
	_plot_metric_comparison(
		results,
		"f1_macro",
		project_root / "docs" / "optimization" / "f1_comparison.png",
		"F1 comparison",
	)

	_plot_metrics_evolution(
		project_root / "results" / "test_metrics.json",
		final_metrics,
		project_root / "docs" / "results" / "metrics_evolution.png",
	)

	# Confusion matrix and misclassified analysis
	test_images_dir = _find_test_images_dir(project_root, data_yaml)
	labels_dir = test_images_dir.parent / "labels"

	model = YOLO(str(best_model_path))
	matrix, misclassified = _build_confusion_and_misclassified(
		model=model,
		test_images_dir=test_images_dir,
		labels_dir=labels_dir,
		class_names=class_names,
		imgsz=int(base_cfg["imgsz"]),
		device=str(args.device),
		conf=args.conf,
		iou_threshold=args.iou,
		max_examples=5,
	)

	confusion_path = project_root / "docs" / "confusion_matrix.png"
	_plot_confusion_matrix(matrix, class_names, confusion_path)

	misclassified_json = project_root / "results" / "misclassified_examples.json"
	for item in misclassified:
		item["image"] = _relative_path(project_root, Path(item["image"]))
	_save_json(misclassified_json, {"examples": misclassified})

	misclassified_md = project_root / "docs" / "optimization" / "misclassified_examples.md"
	_save_misclassified_report(project_root, misclassified, misclassified_md)

	# Save optimized config
	optimized_cfg = cfg.copy()
	optimized_cfg["selected_experiment"] = {
		"name": best_result.name,
		"accuracy": best_result.accuracy,
		"f1_macro": best_result.f1_macro,
		"onnx_model": _relative_path(project_root, onnx_path),
	}
	_save_yaml(config_path, optimized_cfg)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
