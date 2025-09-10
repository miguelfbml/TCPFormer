import torch
from ultralytics import YOLO
import os

print("🔍 ANALYZING YOLO MODEL QUALITY")
print("=" * 60)

model_path = "../runs/pose/model3/best.pt"

if os.path.exists(model_path):
    print(f"📁 Loading model: {model_path}")
    
    try:
        model = YOLO(model_path)
        
        # Get model info
        print(f"Model type: {model.model_name if hasattr(model, 'model_name') else 'Unknown'}")
        
        # Check model size/parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Guess model size based on parameters
        if total_params < 5_000_000:
            size = "nano (n)"
            recommendation = "Try YOLOv11s or YOLOv11m for better accuracy"
        elif total_params < 15_000_000:
            size = "small (s)" 
            recommendation = "Try YOLOv11m or YOLOv11l for better accuracy"
        elif total_params < 30_000_000:
            size = "medium (m)"
            recommendation = "Try YOLOv11l or YOLOv11x for best accuracy"
        elif total_params < 60_000_000:
            size = "large (l)"
            recommendation = "Try YOLOv11x for maximum accuracy"
        else:
            size = "extra large (x)"
            recommendation = "You're using the largest model"
        
        print(f"Estimated model size: {size}")
        print(f"Recommendation: {recommendation}")
        
        # Check training metrics if available
        try:
            # Try to load training results
            results_dir = os.path.dirname(model_path)
            results_file = os.path.join(results_dir, "results.csv")
            
            if os.path.exists(results_file):
                print(f"\n📊 Training results found: {results_file}")
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 1:
                    # Show last few lines (final metrics)
                    print("Final training metrics:")
                    headers = lines[0].strip().split(',')
                    final_values = lines[-1].strip().split(',')
                    
                    for header, value in zip(headers, final_values):
                        if 'mAP' in header or 'precision' in header or 'recall' in header:
                            try:
                                print(f"  {header}: {float(value):.3f}")
                            except:
                                print(f"  {header}: {value}")
        except Exception as e:
            print(f"Could not read training results: {e}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

else:
    print(f"❌ Model not found: {model_path}")
    
    # Look for available models
    print("\nLooking for available models...")
    search_paths = [
        "../runs/pose/",
        "runs/pose/",
        "../runs/detect/",
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"\nFound models in {search_path}:")
            for subdir in os.listdir(search_path):
                subpath = os.path.join(search_path, subdir)
                if os.path.isdir(subpath):
                    weights_path = os.path.join(subpath, "weights", "best.pt")
                    if os.path.exists(weights_path):
                        print(f"  - {subdir}/weights/best.pt")

print("\n💡 SOLUTIONS FOR BETTER YOLO ACCURACY:")
print("1. **Use larger model**: YOLOv11l or YOLOv11x instead of smaller variants")
print("2. **Better pre-trained weights**: Use models trained on COCO pose dataset") 
print("3. **Fine-tuning**: Train on similar indoor/motion capture data")
print("4. **Ensemble**: Average predictions from multiple models")
print("5. **Post-processing**: Apply temporal smoothing and outlier removal")