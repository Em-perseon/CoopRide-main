"""
Entry point for CoopRide LLM Weight Optimization.

Usage:
    cd coopride_llm
    python main.py --config configs/coopride_config.yaml
"""

import os
import sys
import argparse
import yaml

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="CoopRide LLM Weight Optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/coopride_config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)

    if not os.path.exists(config_path):
        print("[ERROR] Config file not found: {}".format(config_path))
        sys.exit(1)

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("[INFO] Loaded config from: {}".format(config_path))

    # Dispatch to runner
    from runner.llm_weight_optim_runner import run_training_loop
    run_training_loop(config)


if __name__ == "__main__":
    main()
