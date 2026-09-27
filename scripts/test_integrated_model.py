import argparse
from integrated_inference import OnionInspector

def main():
    parser = argparse.ArgumentParser(description="Test Final Integrated Onion Inspection System")
    parser.add_argument('--source', type=str, required=True, help="Path to input image")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    print("Initializing Integrated Pipeline...")
    inspector = OnionInspector()
    
    print(f"Running inspection on {args.source}...")
    results, img_disp, out_path = inspector.inspect(source=args.source, conf_threshold=args.conf)
    
    inspector.print_report(results)
    print(f"\nAnnotated image saved to: {out_path}")

if __name__ == "__main__":
    main()
