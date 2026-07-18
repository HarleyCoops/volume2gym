import json
from pathlib import Path
from typing import List, Dict

def merge_results(output_dir: str = "RailroadEngineer1959/data/railroad_extracted"):
    """
    Merge all partial JSON files from parallel extraction into single master files.
    """
    base_path = Path(output_dir)
    
    all_rules = []
    all_tasks = []
    
    # Find all partial files (e.g., railroad_rules_part_1.json)
    rule_files = sorted(base_path.glob("railroad_rules_*.json"))
    task_files = sorted(base_path.glob("safety_tasks_*.json"))
    
    print(f"Found {len(rule_files)} rule files and {len(task_files)} task files.")
    
    # Merge Rules
    seen_rule_ids = set()
    for rf in rule_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
                for rule in data:
                    # Deduplicate by rule_id just in case
                    if rule['rule_id'] not in seen_rule_ids:
                        all_rules.append(rule)
                        seen_rule_ids.add(rule['rule_id'])
        except Exception as e:
            print(f"Error reading {rf}: {e}")

    # Merge Tasks
    seen_task_ids = set()
    for tf in task_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
                for task in data:
                    if task['task_id'] not in seen_task_ids:
                        all_tasks.append(task)
                        seen_task_ids.add(task['task_id'])
        except Exception as e:
            print(f"Error reading {tf}: {e}")
            
    # Sort by page number if available, else rule_id
    all_rules.sort(key=lambda x: (x.get('page_number', 0), x.get('rule_id', '')))
    all_tasks.sort(key=lambda x: x.get('task_id', ''))

    # Save Master Files
    master_rules = base_path / "railroad_rules_complete.json"
    master_tasks = base_path / "safety_tasks_complete.json"
    
    with open(master_rules, "w", encoding="utf-8") as f:
        json.dump(all_rules, f, indent=2, ensure_ascii=False)
        
    with open(master_tasks, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)
        
    print(f"\nMERGE COMPLETE:")
    print(f"- Rules: {len(all_rules)} saved to {master_rules}")
    print(f"- Tasks: {len(all_tasks)} saved to {master_tasks}")

if __name__ == "__main__":
    merge_results()
