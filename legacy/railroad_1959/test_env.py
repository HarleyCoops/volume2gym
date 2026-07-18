from RailroadEngineer1959.railroad_rl_training.verifiers.railroad_env import RailroadEnv
import os

def test_env():
    print("Initializing Railroad Environment...")
    # Point to the extracted tasks we just generated
    tasks_path = "RailroadEngineer1959/data/railroad_extracted/safety_tasks.json"
    
    if not os.path.exists(tasks_path):
        print(f"Tasks file not found at {tasks_path}. Please run extraction first.")
        return

    env = RailroadEnv(tasks_path=tasks_path)
    
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Observation:\n{obs}")
    print(f"Info: {info}")
    
    # Simulate an agent response
    action = "I would ensure the train is stopped and protection is provided as per Rule 99."
    print(f"\nAgent Action: {action}")
    
    print("\nStepping environment...")
    # This will call the LLM rubric, so it might take a moment
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Reward: {reward}")
    print(f"Info: {info}")

if __name__ == "__main__":
    test_env()
