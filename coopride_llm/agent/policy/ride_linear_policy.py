"""
CoopRide Linear Dispatch Policy

10-dimensional feature weight vector for ride-hailing dispatch optimization.
Interface aligned with llm4jssp's JSSPLinearPolicy.

Feature list (10 dims):
    0  dest_supply_shortage   - Destination driver shortage (End - Start)
    1  order_price            - Order fare
    2  duration_cost          - Trip duration penalty
    3  price_efficiency       - Fare per minute efficiency
    4  real_order_bias        - Real order bias (1 if real, 0 if virtual)
    5  current_grid_push      - Current grid congestion push (local entropy)
    6  dest_order_pull        - Destination order pull (order increment)
    7  cross_grid_penalty     - Cross-grid dispatch penalty
    8  global_imbalance       - Global supply-demand imbalance
    9  mdp_advantage          - Long-term value (V-value from RL)
"""

import torch
import numpy as np


N_FEATURES = 10

FEATURE_NAMES = [
    'dest_supply_shortage', 'order_price', 'duration_cost',
    'price_efficiency', 'real_order_bias', 'current_grid_push',
    'dest_order_pull', 'cross_grid_penalty', 'global_imbalance',
    'mdp_advantage'
]

# Typical sign conventions for each feature weight
FEATURE_SIGNS = [
    '+',   # w0: shortage -> positive encourages going to driver-scarce areas
    '+',   # w1: price -> positive incentivizes high-value orders
    '-',   # w2: duration -> negative penalizes long trips
    '+',   # w3: efficiency -> positive favors high efficiency
    '+',   # w4: real order -> positive prioritizes real orders
    '+',   # w5: push -> positive pushes away from congested grids
    '+',   # w6: pull -> positive attracts to order-dense areas
    '-',   # w7: cross-grid -> negative penalizes cross-grid dispatch
    '+',   # w8: imbalance -> positive corrects global imbalance
    '+',   # w9: MDP advantage -> positive considers future value
]


def extract_features(state, order, mask):
    """
    Extract 10-dimensional feature tensor from environment state and order tensors.

    Parameters
    ----------
    state : torch.Tensor, shape=(Grid_Num, State_Dim)
        Grid-level state features.
    order : torch.Tensor, shape=(Grid_Num, Max_Order_Num, Order_Dim)
        Per-order features.
    mask  : torch.Tensor, shape=(Grid_Num, Max_Order_Num)
        Valid order mask.

    Returns
    -------
    torch.Tensor, shape=(Grid_Num, Max_Order_Num, 10)
    """
    grid_num, max_order_num, _ = order.shape
    epsilon = 1e-6

    # --- Tier 1: Decisive factors ---

    # F0: Destination Supply Shortage (driver diff: End - Start)
    # order[..., 6] is diff; higher means more drivers at dest (not scarce).
    # We want "scarcity", so negate.
    f0_shortage = -order[..., 6]

    # F1: Order Price
    f1_price = order[..., 2]

    # F2: Duration Cost
    f2_duration = order[..., 3]

    # F3: Price Efficiency (fare per minute)
    f3_efficiency = f1_price / (f2_duration + epsilon)

    # --- Tier 2: Policy adjustment factors ---

    # F4: Real Order Bias (service_type == -1 means real order)
    f4_real_bias = (order[..., 4] == -1).float()

    # F5: Current Grid Push (local entropy: idle / (real + idle))
    f5_push = state[..., 4].unsqueeze(-1).expand(-1, max_order_num)

    # F6: Destination Order Pull (order num diff: End - Start)
    f6_pull = order[..., 7]

    # --- Tier 3: Auxiliary factors ---

    # F7: Cross-Grid Penalty (start_grid != end_grid)
    f7_cross = (order[..., 0] != order[..., 1]).float()

    # F8: Global Imbalance (abs(local_entropy - global_entropy))
    f8_imbalance = state[..., 5].unsqueeze(-1).expand(-1, max_order_num)

    # F9: MDP Advantage (V-value)
    f9_advantage = order[..., -1]

    # Stack features -> (Grid_Num, Max_Order_Num, 10)
    features = torch.stack([
        f0_shortage, f1_price, f2_duration, f3_efficiency,
        f4_real_bias, f5_push, f6_pull, f7_cross,
        f8_imbalance, f9_advantage
    ], dim=-1)

    return features


class RideLinearPolicy:
    """
    Linear dispatch priority policy for ride-hailing.

    Interface aligned with llm4jssp's JSSPLinearPolicy:
    - weight: shape=(10,) — one weight per feature, no bias
    - compute_scores(): returns priority logits
    - update_policy(): load from flat parameter list
    - get_parameters(): return flat parameter vector
    """

    def __init__(self, weight_range=(-5.0, 5.0), weight_precision=4):
        self.dim_states = N_FEATURES   # 10
        self.weight_range = weight_range
        self.weight_precision = weight_precision
        self.weight = torch.zeros(self.dim_states)

    def initialize_policy(self):
        """Random initialization (uniform in weight_range)."""
        low, high = self.weight_range
        self.weight = torch.round(
            torch.empty(self.dim_states).uniform_(low, high),
            decimals=self.weight_precision
        )

    def compute_scores(self, state, order, mask, device='cpu'):
        """
        Compute linear correction logits: Score_LLM = features @ weights.

        Parameters
        ----------
        state : torch.Tensor, shape=(Grid_Num, State_Dim)
        order : torch.Tensor, shape=(Grid_Num, Max_Order_Num, Order_Dim)
        mask  : torch.Tensor, shape=(Grid_Num, Max_Order_Num)
        device : str

        Returns
        -------
        torch.Tensor, shape=(Grid_Num, Max_Order_Num)
        """
        features = extract_features(state, order, mask)  # (N, K, 10)
        w = self.weight.to(device)
        correction_logits = torch.matmul(features.to(device), w)  # (N, K)

        if mask is not None:
            correction_logits = correction_logits * mask.float().to(device)

        return correction_logits

    def update_policy(self, weight_list):
        """Load weights from a flat parameter list."""
        if weight_list is None:
            return
        if isinstance(weight_list, (list, np.ndarray)):
            self.weight = torch.tensor(weight_list, dtype=torch.float32)
        elif isinstance(weight_list, torch.Tensor):
            self.weight = weight_list.float()
        # Clamp to valid range
        low, high = self.weight_range
        self.weight = self.weight.clamp(low, high)

    def get_parameters(self):
        """Return flat parameter vector as list."""
        return self.weight.tolist()

    def __str__(self):
        output = "Weights (10 features, ride-hailing dispatch):\n"
        for name, sign, w in zip(FEATURE_NAMES, FEATURE_SIGNS, self.weight.tolist()):
            output += "  {} ({}): {:.4f}\n".format(name, sign, w)
        return output
