"""
Runner for CoopRide LLM Weight Optimization

Training loop: warmup → LLM optimization iterations → save best weights.
Interface aligned with llm4jssp's runner.
"""

import os
import sys
import torch
import numpy as np

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from agent.llm_weight_optim_agent import LLMWeightOptimAgent
from runner.coopride_evaluator import CoopRideEvaluator


def run_training_loop(config):
    """
    Main training loop for LLM weight optimization.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML.
        Must include simulator section with grid_num.
    """
    # Resolve relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    template_dir = config.get('template_dir', 'agent/policy/templates')
    if not os.path.isabs(template_dir):
        template_dir = os.path.join(base_dir, template_dir)

    template_name = config.get('template_name', 'coopride_num_text.j2')
    template_path = os.path.join(template_dir, template_name)

    env_desc_file = config.get('env_desc_file', None)
    if env_desc_file and not os.path.isabs(env_desc_file):
        env_desc_file = os.path.join(base_dir, env_desc_file)

    logdir = config.get('logdir', 'logs/coopride_llm')
    if not os.path.isabs(logdir):
        logdir = os.path.join(base_dir, logdir)

    # Print configuration
    print("=" * 60)
    print("CoopRide LLM Weight Optimization")
    print("=" * 60)
    print("[Config] model: {}".format(config.get('llm_model_name', 'gpt-4o-mini')))
    print("[Config] signal_mode: {}".format(config.get('signal_mode', 'num_text')))
    print("[Config] template: {}".format(template_name))
    print("[Config] max_steps: {}".format(config.get('max_steps', 40)))
    print("[Config] warmup_episodes: {}".format(config.get('warmup_episodes', 5)))
    print("[Config] logdir: {}".format(logdir))

    # Print simulator config
    sim_config = config.get('simulator', {})
    print("[Config] grid_num: {}".format(sim_config.get('grid_num', 121)))
    print("[Config] sample: {}".format(sim_config.get('sample', False)))
    print("=" * 60)

    # Set random seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create evaluator (lazy initialization on first call)
    evaluator = CoopRideEvaluator(sim_config)
    evaluate_fn = evaluator.evaluate

    # Create agent
    agent = LLMWeightOptimAgent(
        logdir=logdir,
        template_path=template_path,
        model_name=config.get('llm_model_name', 'gpt-4o-mini'),
        api_key=config.get('api_key', None),
        base_url=config.get('base_url', None),
        signal_mode=config.get('signal_mode', 'num_text'),
        env_desc_file=env_desc_file,
        weight_range=tuple(config.get('weight_range', [-5.0, 5.0])),
        weight_precision=config.get('weight_precision', 4),
        max_traj_count=config.get('max_traj_count', 100),
        topK_size=config.get('topK_size', 7),
        max_steps=config.get('max_steps', 40),
        search_step_size=config.get('search_step_size', 1.0),
    )

    # Warmup
    warmup_episodes = config.get('warmup_episodes', 5)
    agent.random_warmup(warmup_episodes, evaluate_fn)

    # LLM optimization iterations
    num_episodes = config.get('num_episodes', 40)
    for episode in range(num_episodes):
        result = agent.train_step(evaluate_fn, step_number=episode)

        if not result['success']:
            print("[WARNING] Iteration {} failed, retrying...".format(episode))
            continue

    # Save best weights
    best_orr_weights = agent.get_best_weights()
    best_gmv_weights = agent.get_best_gmv_weights()

    torch.save(best_orr_weights, os.path.join(logdir, "best_orr_weights.pt"))
    torch.save(best_gmv_weights, os.path.join(logdir, "best_gmv_weights.pt"))

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("Best ORR weights saved to: {}".format(
        os.path.join(logdir, "best_orr_weights.pt")))
    print("Best GMV weights saved to: {}".format(
        os.path.join(logdir, "best_gmv_weights.pt")))
    print("=" * 60)

    return agent
