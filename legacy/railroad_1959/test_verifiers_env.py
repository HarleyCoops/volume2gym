import os
import sys
from pathlib import Path

# Add the environment package to path so we can import it
sys.path.append(str(Path(__file__).parent / "environments" / "railroad_1959"))

from railroad_1959 import load_environment

def test_verifiers_env():
    print("Initializing Railroad 1959 Environment (Verifiers)...")
    
    # Path to the merged tasks
    tasks_path = Path("RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json")
    
    if not tasks_path.exists():
        print(f"Tasks file not found at {tasks_path}")
        return

    env = load_environment(dataset_path=tasks_path, max_examples=5)
    
    print(f"\nLoaded environment with {len(env.dataset)} examples.")
    
    print(f"\nLoaded environment with {len(env.dataset)} examples.")
    
    # Get a sample task
    sample = env.dataset[0]
    print(f"\nSample Task:\n{sample['question']}")
    print(f"Expected Outcome:\n{sample['answer']}")
    
    # Test the rubric directly
    print("\nTesting Rubric...")
    
    # Mock a completion object (verifiers expects a list of objects with .role and .content)
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content
            
    completion = [
        Message("user", sample['question']),
        Message("assistant", "I would ensure the train is stopped and protection is provided as per Rule 99.")
    ]
    
    # Score
    reward = env.rubric.score(
        completion=completion,
        answer=sample['answer'],
        info=sample['info']
    )
    
    print(f"Reward: {reward}")
    
    ledger = env.rubric.get_last_ledger()
    if ledger:
        print("\nLedger:")
        for k, v in ledger.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    test_verifiers_env()
