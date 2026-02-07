import argparse
from pathlib import Path
from ultralytics import YOLO


def export_to_onnx(
    model_path: str,
    output_dir: str = None,
    imgsz: int = 640,
    batch: int = 1,
    simplify: bool = True,
    dynamic: bool = False,
    opset: int = 12
):
    """
    Export YOLOv8 model to ONNX format
    
    Args:
        model_path: Path to the trained .pt model
        output_dir: Directory to save the ONNX model (default: same as model)
        imgsz: Input image size
        batch: Batch size for export
        simplify: Simplify the ONNX model
        dynamic: Enable dynamic input shapes
        opset: ONNX opset version
    """
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Export to ONNX
    print(f"\nExporting to ONNX format...")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Simplify: {simplify}")
    print(f"  Dynamic shapes: {dynamic}")
    print(f"  ONNX opset: {opset}")
    
    export_path = model.export(
        format='onnx',
        imgsz=imgsz,
        batch=batch,
        simplify=simplify,
        dynamic=dynamic,
        opset=opset
    )
    
    print(f"\n✓ Model exported successfully!")
    print(f"  ONNX model saved at: {export_path}")
    
    # If output directory is specified, move the file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        export_path = Path(export_path)
        dest_path = output_dir / export_path.name
        shutil.move(str(export_path), str(dest_path))
        print(f"  Moved to: {dest_path}")
        return str(dest_path)
    
    return export_path


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 model to ONNX format')
    parser.add_argument(
        '--model',
        type=str,
        default='models/trained_model_v1.pt',
        help='Path to trained model (.pt file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for ONNX model (default: same as model)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Batch size for export (default: 1)'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX model simplification'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic input shapes'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='ONNX opset version (default: 12)'
    )
    
    args = parser.parse_args()
    
    # Export the model
    export_to_onnx(
        model_path=args.model,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        opset=args.opset
    )


if __name__ == '__main__':
    main()
