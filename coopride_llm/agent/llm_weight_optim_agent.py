"""
LLM Weight Optimization Agent for CoopRide

Orchestrates policy, replay buffer, and LLM brain for iterative
weight optimization. Interface aligned with llm4jssp's LLMNumOptimJSSPAgent.
"""

import os
import time
import numpy as np
import torch

from agent.policy.ride_linear_policy import RideLinearPolicy, N_FEATURES, FEATURE_NAMES
from agent.policy.replay_buffer import RideRewardBuffer
from agent.policy.llm_brain import LLMBrain


class LLMWeightOptimAgent:
    """
    Agent that uses LLM to iteratively optimize 10-dimensional dispatch weights.

    Workflow:
    1. Warmup: random weight sampling + evaluation
    2. Iteration: buffer.format_for_llm() → brain.call_llm() → policy.update_policy()
    3. Evaluation: run dispatch with new weights → record ORR/GMV
    """

    def __init__(
        self,
        logdir,
        template_path,
        model_name,
        api_key=None,
        base_url=None,
        signal_mode="num_text",
        env_desc_file=None,
        weight_range=(-5.0, 5.0),
        weight_precision=4,
        max_traj_count=100,
        topK_size=7,
        max_steps=40,
        search_step_size=1.0,
    ):
        self.logdir = logdir
        self.signal_mode = signal_mode
        self.env_desc_file = env_desc_file
        self.max_steps = max_steps
        self.search_step_size = search_step_size
        self.topK_size = topK_size

        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_iterations = 0

        # Policy
        self.policy = RideLinearPolicy(
            weight_range=weight_range,
            weight_precision=weight_precision
        )
        self.rank = N_FEATURES  # 10

        # Replay Buffer
        self.replay_buffer = RideRewardBuffer(max_size=max_traj_count)

        # LLM Brain
        self.llm_brain = LLMBrain(
            template_path=template_path,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )

        # Ensure log directory
        os.makedirs(logdir, exist_ok=True)

    def random_warmup(self, num_warmup, evaluate_fn):
        """
        Random warmup phase: sample random weights and evaluate.

        Parameters
        ----------
        num_warmup : int
            Number of random weight configurations to try.
        evaluate_fn : callable
            Function that takes a weight tensor and returns (orr, gmv).
            Signature: evaluate_fn(weights: torch.Tensor) -> (float, float)
        """
        print("[Warmup] Generating {} random weight configurations...".format(num_warmup))

        for i in range(num_warmup):
            self.policy.initialize_policy()
            weights = self.policy.weight.clone()

            orr, gmv = evaluate_fn(weights)

            self.replay_buffer.add(
                weights=weights.tolist(),
                orr=orr,
                gmv=gmv,
                step_num=i + 1,
                reasoning="Random warmup"
            )

            print("  Warmup {}/{}: ORR={:.4f}, GMV={:.2f}, weights={}".format(
                i + 1, num_warmup, orr, gmv,
                [round(w, 2) for w in weights.tolist()]))

        self.total_iterations = num_warmup

    def train_step(self, evaluate_fn, step_number=None):
        """
        Single LLM optimization step.

        Parameters
        ----------
        evaluate_fn : callable
            Function that takes a weight tensor and returns (orr, gmv).
        step_number : int or None
            Current iteration number. If None, auto-increment.

        Returns
        -------
        dict : {weights, orr, gmv, api_time, success}
        """
        if step_number is None:
            step_number = self.total_iterations + 1

        # Create episode log directory
        episode_dir = os.path.join(self.logdir, "episode_{}".format(step_number))
        os.makedirs(episode_dir, exist_ok=True)

        # Format buffer for LLM
        include_reasoning = (self.signal_mode == "num_text")
        buffer_str = self.replay_buffer.format_for_llm(
            num_params=self.rank,
            include_reasoning=include_reasoning,
            max_entries=self.topK_size
        )

        # Build prompt via Jinja2 template
        template_vars = {
            'rank': self.rank,
            'max_steps': self.max_steps,
            'step_number': step_number,
            'step_size': self.search_step_size,
            'episode_reward_buffer_string': buffer_str,
            'weights': self.policy.get_parameters(),
        }

        # For num_text mode, include env description path (relative to template dir)
        if self.signal_mode == "num_text" and self.env_desc_file:
            template_dir = os.path.dirname(self.llm_brain.template.filename)
            rel_path = os.path.relpath(self.env_desc_file, template_dir)
            # Jinja2 requires forward slashes
            template_vars['env_description'] = rel_path.replace('\\', '/')

        prompt = self.llm_brain.render_prompt(**template_vars)

        # Save prompt to log
        with open(os.path.join(episode_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
            f.write(prompt)

        # Call LLM
        llm_result = self.llm_brain.call_llm(prompt)
        api_time = llm_result.get('api_time', 0)
        self.api_call_time += api_time

        # Save LLM response
        with open(os.path.join(episode_dir, "llm_response.txt"), 'w', encoding='utf-8') as f:
            f.write(llm_result.get('response', ''))

        # Parse weights from response
        new_weights = None
        if llm_result.get('success', False):
            response_text = llm_result['response']
            # Try JSON format first, then params[] format
            new_weights = self.llm_brain.parse_weights(response_text, self.rank)
            if new_weights is None:
                new_weights = self.llm_brain.parse_params_line(response_text, self.rank)

        if new_weights is None:
            print("[WARNING] Failed to parse weights from LLM response at iteration {}".format(
                step_number))
            return {
                'weights': None, 'orr': None, 'gmv': None,
                'api_time': api_time, 'success': False
            }

        # Update policy
        self.policy.update_policy(new_weights)
        weights_tensor = self.policy.weight.clone()

        # Save parameters
        with open(os.path.join(episode_dir, "parameters.txt"), 'w', encoding='utf-8') as f:
            f.write(str(self.policy))

        # Evaluate
        orr, gmv = evaluate_fn(weights_tensor)

        # Record to buffer
        reasoning = llm_result.get('response', '') if include_reasoning else None
        self.replay_buffer.add(
            weights=weights_tensor.tolist(),
            orr=orr,
            gmv=gmv,
            step_num=step_number,
            reasoning=reasoning
        )

        self.total_iterations = step_number

        # Save evaluation result
        with open(os.path.join(episode_dir, "evaluation.txt"), 'w', encoding='utf-8') as f:
            f.write("Iteration: {}\n".format(step_number))
            f.write("ORR: {:.4f}\n".format(orr))
            f.write("GMV: {:.2f}\n".format(gmv))
            f.write("API Time: {:.2f}s\n".format(api_time))

        # Append to overall log
        overall_log = os.path.join(self.logdir, "overall_log.txt")
        header_needed = not os.path.exists(overall_log)
        with open(overall_log, 'a', encoding='utf-8') as f:
            if header_needed:
                f.write("Iteration, CPU Time, API Time, ORR, GMV\n")
            cpu_time = time.process_time() - self.start_time
            f.write("{}, {:.2f}, {:.2f}, {:.4f}, {:.2f}\n".format(
                step_number, cpu_time, self.api_call_time, orr, gmv))

        print("  Iter {}: ORR={:.4f}, GMV={:.2f}, API={:.1f}s".format(
            step_number, orr, gmv, api_time))

        return {
            'weights': weights_tensor,
            'orr': orr,
            'gmv': gmv,
            'api_time': api_time,
            'success': True
        }

    def get_best_weights(self):
        """Return the best weight configuration based on ORR."""
        if not self.replay_buffer.buffer:
            return self.policy.weight.clone()
        best = max(self.replay_buffer.buffer, key=lambda x: x['orr'])
        return torch.tensor(best['weights'])

    def get_best_gmv_weights(self):
        """Return the best weight configuration based on GMV."""
        if not self.replay_buffer.buffer:
            return self.policy.weight.clone()
        best = max(self.replay_buffer.buffer, key=lambda x: x['gmv'])
        return torch.tensor(best['weights'])
