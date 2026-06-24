import argparse
from pathlib import Path

# Yeni yapiya uygun importlar
from Scripts.OCR.pipeline import process_image, run_camera
from Scripts.OCR.schema import to_json

def main():
    parser = argparse.ArgumentParser(description="MRZ OCR pipeline")
    sub = parser.add_subparsers(dest="cmd")

    img_p = sub.add_parser("image", help="Process a still image")
    img_p.add_argument("path", help="Path to image file")
    img_p.add_argument("-o", "--output", help="Write JSON result to this specific file path")
    img_p.add_argument("--output-dir", default="Images/Outputs",
                       help="Save JSON + annotated image to this directory (default: Images/Outputs)")
    img_p.add_argument("--weights", help="Path to YOLO weights (best.pt)")
    img_p.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")

    cam_p = sub.add_parser("camera", help="Live camera scan")
    cam_p.add_argument("--index", type=int, default=0, help="Camera index")
    cam_p.add_argument("--output-dir", default="Images/Outputs",
                       help="Save JSON result to this directory after scan (default: Images/Outputs)")
    cam_p.add_argument("--weights", help="Path to YOLO weights (best.pt)")
    cam_p.add_argument("--conf", type=float, default=0.45)
    cam_p.add_argument("--no-display", action="store_true", help="Headless mode")

    args = parser.parse_args()

    weights = Path(args.weights) if getattr(args, "weights", None) else None

    if args.cmd == "image":
        output_dir = Path(args.output_dir)
        result = process_image(args.path, weights=weights, conf_threshold=args.conf,
                               output_dir=output_dir)
        out = to_json(result)
        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
            print(f"Saved to {args.output}")
        else:
            # args.path icinde bastaki/sondaki bosluklari sil
            safe_path = str(args.path).strip()
            stem = Path(safe_path).stem
            saved = output_dir / f"{stem}_ocr.json"
            print(out)
            print(f"\nSaved: {saved}")
            if (output_dir / f"{stem}_annotated.jpg").exists():
                print(f"       {output_dir / f'{stem}_annotated.jpg'}")

    elif args.cmd == "camera":
        result = run_camera(
            camera_index=args.index,
            weights=weights,
            conf_threshold=args.conf,
            display=not args.no_display,
        )
        out = to_json(result)
        print(out)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"camera_{ts}_ocr.json"
        json_path.write_text(out, encoding="utf-8")
        print(f"\nSaved: {json_path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()




# python main.py image "Images\MRZ_Data\Processed_data\images\test\2e11ec19-MAR-AS-02002_165552.jpg"
# python main.py camera --index 0



