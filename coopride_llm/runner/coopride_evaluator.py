"""
CoopRide Simulator Evaluator (PPO-Free)

Pure LLM weight evaluation: given a 10-dim weight tensor,
run one episode in the CoopRide simulator and return (orr, gmv).

No PPO/Actor/Critic dependency — implements state processing,
order processing, and dispatch logic directly.

Usage:
    evaluator = CoopRideEvaluator(config)
    orr, gmv = evaluator.evaluate(weights)
"""

import os
import sys
import math
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import float32

# Add project root to path for importing simulator, tools
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.policy.ride_linear_policy import extract_features


# ============================================================
# Standalone state/order processing (replaces PPO methods)
# ============================================================

def load_feature_scope(grid_num, save_dir=None):
    """
    Load feature normalization scope from saved pickle.
    Equivalent to PPO.load_feature_scope().
    """
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, 'save')
    path = os.path.join(save_dir, 'feature_{}.pkl'.format(grid_num))
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        feature = pickle.load(f)
    state = torch.max(torch.abs(feature['state']), dim=0)[0]
    state[state == 0] = 5
    order = torch.max(torch.abs(feature['order']), dim=0)[0]
    order[order == 0] = 5
    return {'state': state, 'order': order}


def process_state(states_node, t, feature_scope=None):
    """
    Convert raw state list to normalized tensor.
    Equivalent to PPO.process_state() with feature_normal=3, no state_diff, no rm_state.

    Parameters
    ----------
    states_node : list of np.ndarray
        Raw state arrays from env.reset() / env.step().
    t : int
        Current time step.
    feature_scope : dict or None
        Normalization scope {'state': tensor, 'order': tensor}.

    Returns
    -------
    torch.Tensor, shape=(grid_num, state_dim)
        Processed state: [time, grid_id, idle_driver, real_order, local_entropy, abs_diff, ...]
    """
    s = np.stack(states_node, axis=0)

    # feature_normal=3: divide by scope
    if feature_scope is not None:
        state_scope = feature_scope['state'].numpy()
        feature_dim = s.shape[1] - 1
        scope_dim = state_scope.shape[0] - 1
        if feature_dim > scope_dim:
            pad = np.ones((feature_dim - scope_dim,), dtype=state_scope.dtype)
            state_scope = np.concatenate([state_scope, pad], axis=0)
        s[:, 1:] /= state_scope[None, 1:]

    # Prepend time column
    time_col = np.zeros((s.shape[0], 1), dtype=float)
    time_col[:, 0] = t
    state = np.concatenate([time_col, s], axis=1)
    return torch.Tensor(state)


def process_order(order_state, t, grid_num, max_order_num, env,
                  feature_scope=None, new_order_entropy=True):
    """
    Convert raw order list to padded tensor + mask, add entropy features, normalize.
    Equivalent to PPO.process_order() with use_mdp=0, use_order_time=False,
    new_order_entropy=True, feature_normal=3.

    Execution order matches PPO exactly:
    1. Padding to fixed-size tensor + mask
    2. add_new_entropy (if enabled) → adds 3 extra columns
    3. feature_normal=3 normalization

    Parameters
    ----------
    order_state : list of list
        Per-grid order lists from env.
        Each order: [begin_node, end_node, price, duration, service_type, entropy]
    t : int
        Current time step (unused when use_order_time=False).
    grid_num : int
        Number of grids.
    max_order_num : int
        Max orders per grid.
    env : CityReal
        Environment (needed for add_new_entropy).
    feature_scope : dict or None
        Normalization scope.
    new_order_entropy : bool
        Whether to add entropy-based features (default True, matching PPO config).

    Returns
    -------
    tuple : (order, mask)
        order: torch.Tensor, shape=(grid_num, max_order_num, 6 or 8)
        mask: torch.Tensor, shape=(grid_num, max_order_num), bool
    """
    order_dim = 6  # use_mdp=0 → 6 dims

    trimmed_state = []
    order_num = []
    for i in range(len(order_state)):
        cur_state = order_state[i]
        if len(cur_state) > max_order_num:
            cur_state = cur_state[:max_order_num]
        trimmed_state.append(cur_state)
        order_num.append(len(cur_state))

    order = torch.zeros((grid_num, max_order_num, order_dim), dtype=float32)
    mask = torch.zeros((grid_num, max_order_num), dtype=torch.bool)
    for i in range(len(trimmed_state)):
        if order_num[i] > 0:
            data = torch.Tensor(trimmed_state[i])
            # Ensure we only take up to order_dim columns
            cols = min(data.shape[1], order_dim)
            order[i, :order_num[i], :cols] = data[:, :cols]
        mask[i, :order_num[i]] = 1

    # Step 2: add_new_entropy (before normalization, matching PPO order)
    if new_order_entropy:
        order = _add_new_entropy(env, order)

    # Step 3: feature_normal=3 normalization (after entropy features added)
    if feature_scope is not None:
        order_scope = feature_scope['order']
        contin_dim = 2  # use_order_time=False → first 2 cols are grid IDs
        # Ensure scope covers all columns (may differ if entropy columns added)
        actual_cols = order.shape[2] - contin_dim
        scope_cols = order_scope.shape[0] - contin_dim
        if actual_cols <= scope_cols:
            order[:, :, contin_dim:] /= order_scope[None, None, contin_dim:contin_dim + actual_cols]
        else:
            # Scope doesn't cover extra columns — normalize what we can
            order[:, :, contin_dim:contin_dim + scope_cols] /= order_scope[None, None, contin_dim:]

    return order, mask


def _add_new_entropy(env, order):
    """
    Add entropy-based features to order tensor.
    Equivalent to PPO.add_new_entropy().

    Adds 3 extra columns:
    - order[..., 5]: entropy_diff (end_entropy - start_entropy)
    - order[..., 6]: driver_num_diff (end - start)
    - order[..., 7]: order_num_diff (end - start)

    Parameters
    ----------
    env : CityReal
        Environment with nodes.
    order : torch.Tensor, shape=(grid_num, max_order_num, 6)

    Returns
    -------
    torch.Tensor, shape=(grid_num, max_order_num, 8)
    """
    driver_num = torch.Tensor([node.idle_driver_num for node in env.nodes]) + 1e-5
    order_num = torch.Tensor([node.real_order_num for node in env.nodes]) + 1e-5
    driver_order = torch.stack([driver_num, order_num], dim=1)
    ORR_entropy = torch.min(driver_order, dim=1)[0] / torch.max(driver_order, dim=1)[0]

    node = order[:, :, :2].long()
    entropy_feature = ORR_entropy[node[:, :, 1]] - ORR_entropy[node[:, :, 0]]
    driver_num_feature = driver_num[node[:, :, 1]] - driver_num[node[:, :, 0]]
    order_num_feature = order_num[node[:, :, 1]] - order_num[node[:, :, 0]]

    order[:, :, 5] = entropy_feature
    order = torch.cat([order, driver_num_feature[:, :, None], order_num_feature[:, :, None]], -1)
    return order


def remove_order_grid(order, keep_grid=True):
    """
    Optionally zero out grid ID columns. Default: keep_grid=True (order_grid=True).
    Equivalent to PPO.remove_order_grid() with order_grid=True.
    """
    if keep_grid:
        return order
    else:
        order[:, :, :2] = 0
        return order


def mask_fake_orders(order, mask, remove_fake=False):
    """
    Optionally mask fake orders. Default: remove_fake=False.
    Equivalent to PPO.mask_fake() with remove_fake_order=False.
    """
    if not remove_fake:
        return mask
    else:
        return mask & (order[:, :, 4] < 0)


def dispatch_local(logits, mask, order_idx, env, sample=False):
    """
    Local dispatch: per-driver greedy/sample selection.
    Equivalent to MAPPO.action() local mode dispatch logic.

    Parameters
    ----------
    logits : torch.Tensor, shape=(grid_num, max_order_num)
        Priority scores for each order.
    mask : torch.Tensor, shape=(grid_num, max_order_num), bool
        Valid order mask.
    order_idx : list of list
        Per-grid order ID lists.
    env : CityReal
        Environment (for reading idle_driver_num).
    sample : bool
        True for stochastic, False for greedy.

    Returns
    -------
    action_ids : list of list
        Per-grid selected order IDs.
    """
    grid_num = logits.shape[0]
    max_order_num = logits.shape[1]
    action_ids = []

    for i in range(grid_num):
        driver_num = env.nodes[i].idle_driver_num
        driver_num = min(driver_num, max_order_num)

        if driver_num == 0 or len(order_idx[i]) == 1:
            choose = [0]  # stay local
        else:
            choose = []
            logit = logits[i][mask[i]].clone()
            prob = F.softmax(logit, dim=-1)
            mask_d = mask[i].clone()

            for d in range(driver_num):
                if sample:
                    choose.append(torch.multinomial(prob, 1, replacement=True).item())
                else:
                    choose.append(torch.argmax(prob).item())

                # Remove selected order from pool (unless it's the "stay local" option)
                if choose[-1] > 0:
                    mask_d[choose[-1]] = 0
                    logit[choose[-1]] = -math.inf
                    prob = F.softmax(logit, dim=-1)

                # If stay-local probability reaches 1, no more useful orders
                if prob[0] == 1:
                    break

        action_ids.append([order_idx[i][idx] for idx in choose])

    return action_ids


# ============================================================
# Main evaluator class
# ============================================================

class CoopRideEvaluator:
    """
    Wraps the CoopRide simulator for pure LLM weight evaluation.

    No PPO/Actor/Critic dependency. Implements state processing,
    order processing, and dispatch logic directly.

    Initialization:
    1. Creates environment (DiDi121 / NYU143)
    2. Loads feature normalization scope
    3. Provides evaluate(weights) -> (orr, gmv)

    The evaluate function runs one full episode (144 time steps = 1 day)
    using the given linear weights as the dispatch policy.
    """

    # Simulation constants
    DISPATCH_INTERVAL = 10  # minutes
    TIME_LEN = 144          # 1440 / 10 = 144 steps per day

    # Max order nums per grid
    MAX_ORDER_DICT = {100: 60, 121: 100, 143: 100}

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Must contain:
            - grid_num: int (121 or 143)
            Optional:
            - driver_num: int (default: auto from grid_num)
            - env_seed: int (default: auto from grid_num)
            - device: str ('cpu' or 'gpu', default: 'cpu')
            - sample: bool (True for stochastic, False for greedy, default: False)
            - eval_seed: int (random seed for evaluation, default: 1314520)
        """
        self.config = config
        self.grid_num = config['grid_num']

        # Auto-fill defaults
        driver_dict = {143: 2000, 121: 1500, 100: 1000}
        seed_dict = {143: 326, 121: 6, 100: 16}
        self.driver_num = config.get('driver_num', driver_dict.get(self.grid_num, 1000))
        self.env_seed = config.get('env_seed', seed_dict.get(self.grid_num, 0))
        self.sample = config.get('sample', False)
        self.eval_seed = config.get('eval_seed', 1314520)
        self.max_order_num = self.MAX_ORDER_DICT.get(self.grid_num, 100)

        device_str = config.get('device', 'cpu')
        self.device = torch.device('cuda' if device_str == 'gpu' and torch.cuda.is_available() else 'cpu')

        # Deferred initialization
        self.env = None
        self.feature_scope = None
        self._initialized = False

    def _lazy_init(self):
        """Initialize environment and feature scope on first call."""
        if self._initialized:
            return

        from tools.load_data import load_envs_DiDi121, load_envs_NYU143
        from tools.create_envs import create_OD

        # Create environment
        if self.grid_num == 121:
            env, M, N, _, grid_num = load_envs_DiDi121(driver_num=self.driver_num)
        elif self.grid_num == 143:
            env, M, N, _, grid_num = load_envs_NYU143(driver_num=self.driver_num)
        elif self.grid_num == 100:
            env, M, N, _, grid_num = create_OD()
        else:
            raise ValueError("Unsupported grid_num: {}".format(self.grid_num))

        env.fleet_help = False  # Pure LLM mode: local dispatch only
        self.env = env

        # Load feature normalization scope (feature_normal=3)
        self.feature_scope = load_feature_scope(self.grid_num)
        if self.feature_scope is None:
            print("[Evaluator] WARNING: feature scope not found, running without normalization")

        self._initialized = True
        print("[Evaluator] Initialized: grid={}, max_order={}".format(
            self.grid_num, self.max_order_num))

    def evaluate(self, weights):
        """
        Run one episode with the given weights and return (orr, gmv).

        Parameters
        ----------
        weights : torch.Tensor, shape=(10,)
            Linear dispatch policy weights.

        Returns
        -------
        tuple : (orr, gmv)
            orr: float, order response rate (0-1)
            gmv: float, gross merchandise volume
        """
        self._lazy_init()

        env = self.env
        w = weights.float()

        # Set seed for reproducibility (must use env.reset_randomseed to set
        # RANDOM_SEED which controls per-step order generation seeds)
        env.reset_randomseed(self.env_seed * 1000)

        # Reset environment
        states_node, _, order_states, order_idx, order_feature, _ = env.reset(mode='PPO2')

        # Process initial state and orders
        state = process_state(states_node, 0, self.feature_scope)
        order, mask_order = process_order(
            order_states, 0, self.grid_num, self.max_order_num, env,
            self.feature_scope)
        order = remove_order_grid(order, keep_grid=True)
        mask_order = mask_fake_orders(order, mask_order, remove_fake=False)

        gmv_total = 0.0
        order_response_rate = 0.0

        for T in range(self.TIME_LEN):
            # Compute linear logits: features @ weights -> (grid_num, max_order_num)
            features = extract_features(state, order, mask_order)  # (N, K, 10)
            llm_logits = torch.matmul(features, w)  # (N, K)

            # Mask invalid orders with -inf
            llm_logits[~mask_order.bool()] = -1e9

            # Dispatch: assign drivers to orders
            action_ids = dispatch_local(
                llm_logits, mask_order, order_idx, env, sample=self.sample)

            # Execute environment step
            orders = env.get_orders_by_id(action_ids)
            next_states_node, next_order_states, next_order_idx, next_order_feature = \
                env.step(orders, generate_order=1, mode='PPO2')

            # Record metrics
            gmv_total += env.gmv
            if env.order_response_rate >= 0:
                order_response_rate = env.order_response_rate

            # Prepare next state
            state = process_state(next_states_node, T + 1, self.feature_scope)
            order, mask_order = process_order(
                next_order_states, T + 1, self.grid_num, self.max_order_num, env,
                self.feature_scope)
            order = remove_order_grid(order, keep_grid=True)
            mask_order = mask_fake_orders(order, mask_order, remove_fake=False)

            order_idx = next_order_idx

        return order_response_rate, gmv_total
