"""
Tesseract + OCR-B MRZ pipeline entry point.

Usage:
    python main_tess.py image <path>  [--output-dir Images/Outputs] [--weights best.pt]
    python main_tess.py setup         # download ocrb.traineddata model

The Tesseract pipeline uses the same YOLO detector, preprocessor, MRZ parser
and JSON schema as the main OCR pipeline — only the recognition engine differs.
"""
import argparse
from pathlib import Path

from Scripts.Tesseract.pipeline import process_image
from Scripts.OCR.schema import to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Tesseract OCR-B MRZ pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # ── image ──────────────────────────────────────────────────────────────
    img_p = sub.add_parser("image", help="Process a still image")
    img_p.add_argument("path", help="Path to image file")
    img_p.add_argument("-o", "--output", help="Write JSON to this specific file path")
    img_p.add_argument("--output-dir", default="Images/Outputs",
                       help="Save JSON + annotated image here (default: Images/Outputs)")
    img_p.add_argument("--weights", help="Path to YOLO weights (best.pt)")
    img_p.add_argument("--conf", type=float, default=0.5,
                       help="YOLO confidence threshold")

    # ── setup ──────────────────────────────────────────────────────────────
    sub.add_parser("setup", help="Download ocrb.traineddata model")

    args = parser.parse_args()

    if args.cmd == "setup":
        from Scripts.Tesseract.setup_model import download_ocrb_model
        download_ocrb_model()
        return

    if args.cmd == "image":
        weights = Path(args.weights) if getattr(args, "weights", None) else None
        output_dir = Path(args.output_dir)
        result = process_image(
            args.path,
            weights=weights,
            conf_threshold=args.conf,
            output_dir=output_dir,
        )
        out = to_json(result)

        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
            print(f"Saved to {args.output}")
        else:
            stem = Path(str(args.path).strip()).stem
            print(out)
            saved_json = output_dir / f"{stem}_tess_ocr.json"
            saved_img = output_dir / f"{stem}_tess_annotated.jpg"
            print(f"\nSaved: {saved_json}")
            if saved_img.exists():
                print(f"       {saved_img}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()