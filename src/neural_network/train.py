import os
import json
import csv
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class RampDetectionTrainer:
    """
    Trainer class for YOLOv8 ramp detection model.
    Handles training, validation, metrics logging, and visualization.
    """
    
    def __init__(self, project_root="../.."):
        """
        Initialize trainer with project paths.
        
        Args:
            project_root (str): Root path of the project
        """
        self.project_root = Path(project_root)
        self.data_yaml = self.project_root / "data" / "data.yaml"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.docs_dir = self.project_root / "docs"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Get next model version
        self.model_version = self._get_next_version()
        self.model_name = f"trained_model_v{self.model_version}"
        self.model_path = self.models_dir / f"{self.model_name}.pt"
        
        # Training configuration
        self.config = {
            'model_architecture': 'yolov8m',  # Medium size - good balance for RTX 3050 (4GB VRAM)
            'batch_size': 8,  # Adjusted for 4GB GPU memory (was 16 for larger GPUs)
            'epochs': 75,  # With early stopping at patience=20, stops ~60-70 epochs
            'patience': 20,  # Early stopping patience
            'learning_rate': 0.005,  # Conservative for small dataset
            'lr_scheduler': 'cosine',  # Cosine annealing scheduler
            'warmup_epochs': 3,  # Linear warmup to stabilize training
            'device': 0,  # GPU device (0 for first GPU)
            'save_period': 5,  # Save checkpoint every 5 epochs
            'augmentation': {
                'hsv_h': 0.015,  # HSV-Hue variation
                'hsv_s': 0.7,    # HSV-Saturation variation
                'hsv_v': 0.4,    # HSV-Value variation
                'degrees': 20,   # Rotation degrees
                'translate': 0.1,  # X/Y translation (fraction)
                'scale': 0.3,    # Image scale (+/- %)
                'flipud': 0.5,   # Flip upside-down probability
                'fliplr': 0.5,   # Flip left-right probability
                'mosaic': 1.0,   # Mosaic augmentation (YOLO default)
                'perspective': 0.0002,  # Perspective transform
                'shear': 5,      # Shear degrees
            }
        }
        
        print(f"\n{'='*60}")
        print(f"YOLOv8 Ramp Detection Training")
        print(f"Model: {self.model_name}")
        print(f"Version: {self.model_version}")
        print(f"{'='*60}\n")
    
    def _get_next_version(self):
        """Get next available model version number."""
        existing_models = list(self.models_dir.glob("trained_model_v*.pt"))
        if not existing_models:
            return 1
        
        versions = []
        for model in existing_models:
            try:
                version = int(model.stem.split('v')[-1])
                versions.append(version)
            except ValueError:
                continue
        
        return max(versions) + 1 if versions else 1
    
    def train(self):
        """
        Train the YOLOv8 model with configured hyperparameters.
        
        Returns:
            dict: Training results
        """
        print(f"\n[INFO] Starting training with configuration:")
        for key, value in self.config.items():
            if key != 'augmentation':
                print(f"  {key}: {value}")
        
        # Load model
        model = YOLO(f"{self.config['model_architecture']}.pt")
        
        # Train model
        results = model.train(
            data=str(self.data_yaml),
            epochs=self.config['epochs'],
            batch=self.config['batch_size'],
            imgsz=640,  # Standard YOLO input size, fits RTX 3050 with batch=8
            device=self.config['device'],
            patience=self.config['patience'],  # Early stopping
            lr0=self.config['learning_rate'],  # Initial learning rate
            lrf=0.01,  # Final learning rate ratio
            warmup_epochs=self.config['warmup_epochs'],
            save=True,
            save_period=self.config['save_period'],
            project=str(self.project_root / "runs"),
            name=self.model_name,
            exist_ok=False,
            # Augmentation parameters
            hsv_h=self.config['augmentation']['hsv_h'],
            hsv_s=self.config['augmentation']['hsv_s'],
            hsv_v=self.config['augmentation']['hsv_v'],
            degrees=self.config['augmentation']['degrees'],
            translate=self.config['augmentation']['translate'],
            scale=self.config['augmentation']['scale'],
            flipud=self.config['augmentation']['flipud'],
            fliplr=self.config['augmentation']['fliplr'],
            mosaic=self.config['augmentation']['mosaic'],
            perspective=self.config['augmentation']['perspective'],
            shear=self.config['augmentation']['shear'],
            # Other settings
            verbose=True,
            seed=42,  # Reproducibility
        )
        
        return results
    
    def save_model(self, results):
        """
        Save the final trained model to models directory.
        
        Args:
            results: Training results object
        """
        # The best model is usually at: runs/<model_name>/weights/best.pt
        best_model_path = (
            self.project_root / "runs" / self.model_name / "weights" / "best.pt"
        )
        
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, self.model_path)
            print(f"\n[SUCCESS] Model saved to: {self.model_path}")
        else:
            print(f"\n[WARNING] Best model not found at {best_model_path}")
    
    def plot_loss_curves(self):
        """Generate and save training loss curves."""
        results_csv = (
            self.project_root / "runs" / self.model_name / "results.csv"
        )
        
        if not results_csv.exists():
            print(f"\n[WARNING] Results CSV not found at {results_csv}")
            return
        
        try:
            # Read results
            data = np.genfromtxt(
                results_csv,
                delimiter=',',
                dtype=str,
                skip_header=0
            )
            
            # Extract headers and values
            headers = data[0]
            values = data[1:].astype(float)
            
            # Find loss columns
            loss_idx = None
            val_loss_idx = None
            
            for i, header in enumerate(headers):
                header_clean = header.strip()
                if 'train/loss' in header_clean:
                    loss_idx = i
                elif 'val/loss' in header_clean:
                    val_loss_idx = i
            
            if loss_idx is None or val_loss_idx is None:
                # Alternative names
                for i, header in enumerate(headers):
                    header_clean = header.strip()
                    if 'loss' in header_clean.lower() and 'train' in header_clean.lower():
                        loss_idx = i
                    elif 'loss' in header_clean.lower() and 'val' in header_clean.lower():
                        val_loss_idx = i
            
            if loss_idx is not None and val_loss_idx is not None:
                epochs = np.arange(1, len(values) + 1)
                train_loss = values[:, loss_idx]
                val_loss = values[:, val_loss_idx]
                
                # Create figure
                plt.figure(figsize=(12, 6))
                plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
                plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('YOLOv8 Training Loss Curves', fontsize=14, fontweight='bold')
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                curve_path = self.docs_dir / "loss_curve.png"
                plt.savefig(curve_path, dpi=300, bbox_inches='tight')
                print(f"\n[SUCCESS] Loss curves saved to: {curve_path}")
                plt.close()
            else:
                print("\n[WARNING] Could not find loss columns in results CSV")
        
        except Exception as e:
            print(f"\n[ERROR] Failed to plot loss curves: {str(e)}")
    
    def save_training_history(self):
        """Save training history from results.csv to results/training_history.csv."""
        src_csv = self.project_root / "runs" / self.model_name / "results.csv"
        dst_csv = self.results_dir / "training_history.csv"
        
        if src_csv.exists():
            import shutil
            shutil.copy(src_csv, dst_csv)
            print(f"[SUCCESS] Training history saved to: {dst_csv}")
        else:
            print(f"[WARNING] Source results.csv not found at {src_csv}")
    
    def evaluate_test_set(self):
        """
        Evaluate the trained model on test set and compute metrics.
        
        Returns:
            dict: Test metrics
        """
        if not self.model_path.exists():
            print(f"\n[ERROR] Model not found at {self.model_path}")
            return None
        
        print(f"\n[INFO] Evaluating model on test set...")
        
        model = YOLO(str(self.model_path))
        
        # Get test images path
        test_images_dir = self.project_root / "data" / "test" / "images"
        
        if not test_images_dir.exists():
            print(f"[WARNING] Test images directory not found: {test_images_dir}")
            return None
        
        # Evaluate on test set using YOLO's validation
        # Note: This will use YOLO's built-in metrics
        test_yaml = str(self.data_yaml)
        
        try:
            results = model.val(
                data=test_yaml,
                imgsz=640,
                batch=16,
                device=0,
                verbose=True,
            )
            
            # Extract metrics from YOLO validation results
            # Get per-class metrics and calculate macro averages
            precision_list = results.box.p  # List of precisions per class
            recall_list = results.box.r      # List of recalls per class
            
            # Calculate macro-averaged metrics (average across classes)
            test_precision_macro = float(np.mean(precision_list))
            test_recall_macro = float(np.mean(recall_list))
            
            # Calculate F1 score: 2 * (P * R) / (P + R)
            f1_per_class = 2 * (precision_list * recall_list) / (precision_list + recall_list + 1e-6)
            test_f1_macro = float(np.mean(f1_per_class))
            
            # mAP scores
            map50 = float(results.box.map50)
            map_value = float(results.box.map)
            
            # Accuracy approximation (using mAP50 as overall accuracy)
            test_accuracy = map50
            
            metrics = {
                "test_accuracy": test_accuracy,
                "test_f1_macro": test_f1_macro,
                "test_precision_macro": test_precision_macro,
                "test_recall_macro": test_recall_macro,
                "map50": map50,
                "map": map_value,
            }
            
            return metrics
        
        except Exception as e:
            print(f"[WARNING] Error during evaluation: {str(e)}")
            print("Using fallback metrics calculation...")
            return self._compute_fallback_metrics()
    
    def _compute_fallback_metrics(self):
        """Compute fallback metrics when standard evaluation fails."""
        return {
            "test_accuracy": 0.0,
            "test_f1_macro": 0.0,
            "test_precision_macro": 0.0,
            "test_recall_macro": 0.0,
            "map50": 0.0,
            "map": 0.0,
            "note": "Fallback metrics - model evaluation needs validation"
        }
    
    def save_test_metrics(self, metrics):
        """
        Save test metrics to JSON file.
        
        Args:
            metrics (dict): Test metrics dictionary
        """
        if metrics is None:
            metrics = self._compute_fallback_metrics()
        
        metrics_path = self.results_dir / "test_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"[SUCCESS] Test metrics saved to: {metrics_path}")
        print(f"\nTest Metrics Summary:")
        print(f"  Accuracy: {metrics.get('test_accuracy', 0):.4f}")
        print(f"  F1 (Macro): {metrics.get('test_f1_macro', 0):.4f}")
        print(f"  Precision (Macro): {metrics.get('test_precision_macro', 0):.4f}")
        print(f"  Recall (Macro): {metrics.get('test_recall_macro', 0):.4f}")
        print(f"  mAP@50: {metrics.get('map50', 0):.4f}")
        print(f"  mAP: {metrics.get('map', 0):.4f}")
    
    def save_configuration(self):
        """Save training configuration to JSON."""
        config_path = self.results_dir / f"config_{self.model_name}.json"
        
        config_to_save = self.config.copy()
        config_to_save['model_path'] = str(self.model_path)
        config_to_save['training_date'] = datetime.now().isoformat()
        config_to_save['dataset_info'] = {
            'total_images': 236,
            'classes': {
                'rampDown': 119,
                'rampUp': 154,
                'ramps-railing': 729
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"[INFO] Configuration saved to: {config_path}")


def main():
    """Main training pipeline."""
    try:
        # Initialize trainer
        trainer = RampDetectionTrainer(project_root="../..")
        
        # Train model
        print("\n[PHASE 1] Training Model...")
        results = trainer.train()
        
        # Save model
        print("\n[PHASE 2] Saving Model...")
        trainer.save_model(results)
        
        # Generate visualizations
        print("\n[PHASE 3] Generating Visualizations...")
        trainer.plot_loss_curves()
        trainer.save_training_history()
        
        # Evaluate on test set
        print("\n[PHASE 4] Evaluating on Test Set...")
        metrics = trainer.evaluate_test_set()
        trainer.save_test_metrics(metrics)
        
        # Save configuration
        print("\n[PHASE 5] Saving Configuration...")
        trainer.save_configuration()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Model saved at: {trainer.model_path}")
        print(f"Results saved at: {trainer.results_dir}")
        print(f"Visualizations saved at: {trainer.docs_dir}")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
