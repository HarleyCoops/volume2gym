import os
import sys
import subprocess
from pathlib import Path

def main():
    # Add environment package to PYTHONPATH
    env_path = Path(__file__).parent / "environments" / "railroad_1959"
    
    # Construct command
    # We use the prime_rl module from the framework directory
    framework_path = Path(__file__).parent.parent / "prime-rl-framework" / "src"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env_path};{framework_path};{env.get('PYTHONPATH', '')}"
    
    config_path = Path(__file__).parent / "railroad_rl_training" / "train.toml"
    
    cmd = [
        sys.executable,
        "-m",
        "prime_rl.rl",
        "@",
        str(config_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
