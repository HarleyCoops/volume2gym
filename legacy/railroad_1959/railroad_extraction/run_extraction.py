import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from .core.pdf_processor import PDFProcessor
from .core.vlm_extractor import VLMExtractor
from .core.rule_converter import RuleConverter
from .schemas.railroad_rule import RailroadRule
from .schemas.safety_task import SafetyTask

# Load environment variables
root_env = Path(__file__).parent.parent.parent / ".env"
load_dotenv(root_env)

def save_json(data: List[Dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Run Railroad Extraction Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to PDF file")
    parser.add_argument("--images-dir", type=str, help="Directory containing page images")
    parser.add_argument("--output-dir", type=str, default="data/railroad_extracted", help="Output directory")
    parser.add_argument("--pages", type=str, help="Page range (e.g., '1-10' or '41')")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip PDF to image conversion")
    
    parser.add_argument("--partition", type=str, help="Partition identifier (e.g., 'part1')")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output filenames based on partition
    suffix = f"_{args.partition}" if args.partition else ""
    rules_filename = f"railroad_rules{suffix}.json"
    tasks_filename = f"safety_tasks{suffix}.json"
    
    # 1. Image Acquisition
    image_paths = []
    if args.images_dir:
        img_dir = Path(args.images_dir)
        image_paths = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    elif args.pdf:
        if not args.skip_conversion:
            print(f"Converting PDF: {args.pdf}")
            processor = PDFProcessor(args.pdf, str(output_dir / "images"))
            image_paths = processor.convert_to_images()
        else:
            # Assume images are already there
            img_dir = output_dir / "images"
            image_paths = sorted(list(img_dir.glob("*.png")))
    else:
        print("Error: Must provide --pdf or --images-dir")
        return

    # Filter pages if requested
    if args.pages:
        if "-" in args.pages:
            start, end = map(int, args.pages.split("-"))
            # Assuming filenames contain page numbers like page_001.png
            filtered_paths = []
            for p in image_paths:
                try:
                    # Extract number from filename (e.g. page_041.png -> 41)
                    num = int(''.join(filter(str.isdigit, p.stem)))
                    if start <= num <= end:
                        filtered_paths.append(p)
                except:
                    pass
            image_paths = filtered_paths
        else:
            target = int(args.pages)
            filtered_paths = []
            for p in image_paths:
                try:
                    num = int(''.join(filter(str.isdigit, p.stem)))
                    if num == target:
                        filtered_paths.append(p)
                except:
                    pass
            image_paths = filtered_paths

    if not image_paths:
        print("No images found to process.")
        return

    print(f"Processing {len(image_paths)} pages...")

    # 2. Initialize Extractors
    vlm = VLMExtractor()
    converter = RuleConverter()
    
    all_rules = []
    all_tasks = []

    # 3. Processing Loop
    for i, img_path in enumerate(image_paths):
        print(f"\n--- Processing {img_path.name} ({i+1}/{len(image_paths)}) ---")
        
        # Extract page number
        try:
            page_num = int(''.join(filter(str.isdigit, img_path.stem)))
        except:
            page_num = 0
            
        # Extract Rules
        rules = vlm.extract_rules_from_image(str(img_path), page_number=page_num)
        print(f"Extracted {len(rules)} rules.")
        
        for rule in rules:
            all_rules.append(rule.model_dump())
            
            # Generate Tasks
            tasks = converter.convert_to_tasks(rule)
            print(f"Generated {len(tasks)} tasks for rule {rule.rule_id}")
            
            for task in tasks:
                all_tasks.append(task.model_dump())
        
        # Save intermediate results every page to prevent data loss
        rules_path = output_dir / rules_filename
        tasks_path = output_dir / tasks_filename
        save_json(all_rules, rules_path)
        save_json(all_tasks, tasks_path)
        print(f"Saved progress to {rules_path}")

    print(f"\nPipeline Complete!")
    print(f"Saved {len(all_rules)} rules to {rules_path}")
    print(f"Saved {len(all_tasks)} tasks to {tasks_path}")

if __name__ == "__main__":
    main()
