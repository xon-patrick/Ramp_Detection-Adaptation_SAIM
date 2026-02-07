"""
Evaluate YOLOv8 ramp detection model.
Measures detection metrics (mAP, precision, recall, F1) and inference latency.
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO


def _load_data_yaml(data_yaml_path: Path) -> dict:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    with data_yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_dataset_path(path_value: str, yaml_dir: Path, base_path: Path | None = None) -> Path:
    path_value = str(path_value)
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    if base_path is not None:
        return (base_path / path_obj).resolve()
    return (yaml_dir / path_obj).resolve()


def _find_test_images_dir(project_root: Path, data_yaml_path: Path) -> Path:
    data_cfg = _load_data_yaml(data_yaml_path)
    yaml_dir = data_yaml_path.parent
    tried: list[Path] = []

    base_path = None
    if "path" in data_cfg:
        base_path = _resolve_dataset_path(data_cfg["path"], yaml_dir)

    test_value = data_cfg.get("test")
    if test_value:
        test_path = _resolve_dataset_path(test_value, yaml_dir, base_path)
        tried.append(test_path)
        if test_path.exists():
            return test_path

        test_path_from_data = None
        test_value_path = Path(str(test_value))
        if not test_value_path.is_absolute() and test_value_path.parts[:1] == ("..",):
            test_path_from_data = (yaml_dir / Path(*test_value_path.parts[1:])).resolve()
            tried.append(test_path_from_data)
            if test_path_from_data.exists():
                return test_path_from_data

        test_images_path = test_path / "images"
        tried.append(test_images_path)
        if test_images_path.exists():
            return test_images_path

        if test_path_from_data is not None:
            test_images_from_data = test_path_from_data / "images"
            tried.append(test_images_from_data)
            if test_images_from_data.exists():
                return test_images_from_data

        project_test_path = (project_root / test_value).resolve()
        tried.append(project_test_path)
        if project_test_path.exists():
            return project_test_path

        project_test_images = project_test_path / "images"
        tried.append(project_test_images)
        if project_test_images.exists():
            return project_test_images

        data_test_path = (project_root / "data" / test_value).resolve()
        tried.append(data_test_path)
        if data_test_path.exists():
            return data_test_path

        data_test_images = data_test_path / "images"
        tried.append(data_test_images)
        if data_test_images.exists():
            return data_test_images

    candidates = [
        project_root / "test" / "images",
        project_root / "data" / "test" / "images",
    ]
    for candidate in candidates:
        tried.append(candidate)
        if candidate.exists():
            return candidate

    tried_text = "\n".join(f"- {path}" for path in tried)
    raise FileNotFoundError(
        "Could not locate test images directory. Tried:\n" + tried_text
    )


def _find_latest_model(models_dir: Path) -> Path | None:
    models = sorted(models_dir.glob("trained_model_v*.pt"))
    if not models:
        return None
    return models[-1]


def _sync_if_cuda(device: str) -> None:
    if torch.cuda.is_available() and (device == "0" or device.startswith("cuda")):
        torch.cuda.synchronize()


def _measure_latency(
    model: YOLO,
    image_paths: list[Path],
    device: str,
    imgsz: int,
    batch: int,
    warmup: int,
) -> dict:
    if not image_paths:
        return {
            "avg_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "fps": None,
        }

    warmup_images = image_paths[:warmup]
    if warmup_images:
        model.predict(
            source=[str(p) for p in warmup_images],
            imgsz=imgsz,
            device=device,
            batch=min(batch, len(warmup_images)),
            verbose=False,
        )
        _sync_if_cuda(device)

    latencies_ms: list[float] = []

    for i in range(0, len(image_paths), batch):
        batch_paths = image_paths[i : i + batch]
        start = time.perf_counter()
        model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=imgsz,
            device=device,
            batch=len(batch_paths),
            verbose=False,
        )
        _sync_if_cuda(device)
        end = time.perf_counter()

        per_image_ms = (end - start) * 1000.0 / len(batch_paths)
        latencies_ms.extend([per_image_ms] * len(batch_paths))

    avg_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    p95_ms = statistics.quantiles(latencies_ms, n=100)[94]
    fps = 1000.0 / avg_ms if avg_ms > 0 else None

    return {
        "avg_ms": avg_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def evaluate(
    project_root: Path,
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    warmup: int,
) -> dict:
    model = YOLO(str(model_path))

    val_results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=True,
    )

    precision_list = val_results.box.p
    recall_list = val_results.box.r

    f1_per_class = 2 * (precision_list * recall_list) / (precision_list + recall_list + 1e-6)

    metrics = {
        "map50": float(val_results.box.map50),
        "map": float(val_results.box.map),
        "precision_macro": float(precision_list.mean()),
        "recall_macro": float(recall_list.mean()),
        "f1_macro": float(f1_per_class.mean()),
    }

    test_images_dir = _find_test_images_dir(project_root, data_yaml)
    image_paths = sorted(
        [
            *test_images_dir.glob("*.jpg"),
            *test_images_dir.glob("*.jpeg"),
            *test_images_dir.glob("*.png"),
        ]
    )

    latency = _measure_latency(
        model=model,
        image_paths=image_paths,
        device=device,
        imgsz=imgsz,
        batch=batch,
        warmup=warmup,
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "split": "test",
        "image_count": len(image_paths),
        "metrics": metrics,
        "latency": latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 ramp detection model.")
    parser.add_argument("--project-root", default="../..", help="Project root path.")
    parser.add_argument("--model", default="", help="Path to model .pt file.")
    parser.add_argument("--data", default="", help="Path to data.yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Device id or name.")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup images for timing.")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_yaml = Path(args.data).resolve() if args.data else project_root / "data" / "data.yaml"

    if args.model:
        model_path = Path(args.model).resolve()
    else:
        default_model = project_root / "models" / "trained_model_v1.pt"
        if default_model.exists():
            model_path = default_model
        else:
            latest = _find_latest_model(project_root / "models")
            if latest is None:
                raise FileNotFoundError("No model found in models directory. Use --model.")
            model_path = latest

    results = evaluate(
        project_root=project_root,
        model_path=model_path,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        warmup=args.warmup,
    )

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "eval_metrics.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation Summary")
    print("-" * 60)
    print(f"Model: {results['model_path']}")
    print(f"Data:  {results['data_yaml']}")
    print(f"Images: {results['image_count']}")
    print(f"mAP@50: {results['metrics']['map50']:.4f}")
    print(f"mAP:    {results['metrics']['map']:.4f}")
    print(f"F1:     {results['metrics']['f1_macro']:.4f}")
    print(f"Prec:   {results['metrics']['precision_macro']:.4f}")
    print(f"Recall: {results['metrics']['recall_macro']:.4f}")

    latency = results["latency"]
    if latency["avg_ms"] is not None:
        print(f"Avg latency: {latency['avg_ms']:.2f} ms")
        print(f"Median:      {latency['median_ms']:.2f} ms")
        print(f"P95:         {latency['p95_ms']:.2f} ms")
        print(f"FPS:         {latency['fps']:.2f}")

    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
