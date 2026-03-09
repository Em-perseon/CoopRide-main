"""
Replay Buffer for CoopRide LLM Weight Optimization

Stores historical weight configurations and their performance metrics (ORR, GMV).
Supports TopK selection and LLM-friendly formatting.
Interface aligned with llm4jssp's replay_buffer.py.
"""

import numpy as np


class RideRewardBuffer:
    """
    Historical record buffer for ride-hailing dispatch weight optimization.

    Each entry stores:
        - weights: list of 10 floats
        - orr: float, order response rate
        - gmv: float, gross merchandise volume
        - step_num: int, iteration number
        - reasoning: str or None, LLM reasoning text (for num_text mode)
    """

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.buffer = []

    def add(self, weights, orr, gmv, step_num, reasoning=None):
        """
        Add a new record to the buffer.

        Parameters
        ----------
        weights : list or np.ndarray, shape=(10,)
        orr     : float, order response rate
        gmv     : float, gross merchandise volume
        step_num: int, iteration number
        reasoning: str or None, LLM reasoning text
        """
        if isinstance(weights, np.ndarray):
            weights = weights.tolist()
        elif hasattr(weights, 'tolist'):
            weights = weights.tolist()

        entry = {
            'weights': list(weights),
            'orr': float(orr),
            'gmv': float(gmv),
            'step_num': int(step_num),
            'reasoning': reasoning,
        }
        self.buffer.append(entry)

        # Evict oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_topK(self, k=7):
        """
        Return TopK records for LLM context: current trial + top ORR + top GMV.

        Returns
        -------
        list of dict, sorted: [current, top_orr_1, ..., top_gmv_1, ...]
        """
        if not self.buffer:
            return []

        current = [self.buffer[-1]]

        # Top ORR (excluding current)
        history = self.buffer[:-1] if len(self.buffer) > 1 else []
        sorted_by_orr = sorted(history, key=lambda x: x['orr'], reverse=True)
        top_orr = sorted_by_orr[:k // 2]

        # Top GMV (excluding current and already selected)
        selected_steps = {r['step_num'] for r in top_orr}
        remaining = [r for r in history if r['step_num'] not in selected_steps]
        sorted_by_gmv = sorted(remaining, key=lambda x: x['gmv'], reverse=True)
        top_gmv = sorted_by_gmv[:k - len(top_orr)]

        return current + top_orr + top_gmv

    def format_for_llm(self, num_params=10, include_reasoning=False, max_entries=15):
        """
        Format buffer contents for LLM prompt.

        Parameters
        ----------
        num_params : int, number of weight parameters
        include_reasoning : bool, whether to include LLM reasoning text
        max_entries : int, maximum entries to include

        Returns
        -------
        str : formatted text for LLM prompt
        """
        if not self.buffer:
            return "No historical data available yet."

        # Use TopK selection
        records = self.get_topK(k=max_entries)

        if include_reasoning:
            return self._format_with_reasoning(records)
        else:
            return self._format_table(records)

    def _format_table(self, records):
        """Format as Markdown table (for num_only mode)."""
        header = "| Iteration | ORR (Efficiency) | GMV (Revenue) | Weights (w0 to w9) |\n"
        separator = "| :--- | :--- | :--- | :--- |\n"
        rows = ""

        for i, rec in enumerate(records):
            if i == 0:
                tag = "Current"
            elif i <= len(records) // 2:
                tag = "Top ORR"
            else:
                tag = "Top GMV"
            w_str = "[" + ", ".join(["{:.2f}".format(w) for w in rec['weights']]) + "]"
            rows += "| {} ({}) | {:.4f} | {:.2f} | {} |\n".format(
                rec['step_num'], tag, rec['orr'], rec['gmv'], w_str)

        return header + separator + rows

    def _format_with_reasoning(self, records):
        """Format with reasoning history (for num_text mode)."""
        output = "# Complete Training History (with reasoning)\n\n"

        for i, rec in enumerate(records):
            if i == 0:
                tag = "Current Trial"
            elif i <= len(records) // 2:
                tag = "Top ORR Record"
            else:
                tag = "Top GMV Record"

            output += "## Attempt {} (Iteration {}, {})\n".format(i + 1, rec['step_num'], tag)
            output += "**Parameters**: {}\n".format(
                ", ".join(["w{}={:.4f}".format(j, w) for j, w in enumerate(rec['weights'])]))
            output += "**Performance**: ORR={:.4f}, GMV={:.2f}\n".format(rec['orr'], rec['gmv'])

            if rec.get('reasoning'):
                output += "**Reasoning**:\n{}\n".format(rec['reasoning'])

            output += "\n" + "-" * 60 + "\n\n"

        # Summary statistics
        all_orr = [r['orr'] for r in self.buffer]
        all_gmv = [r['gmv'] for r in self.buffer]
        output += "## Summary Statistics\n"
        output += "- Best ORR: {:.4f}\n".format(max(all_orr))
        output += "- Best GMV: {:.2f}\n".format(max(all_gmv))
        output += "- Mean ORR: {:.4f}\n".format(np.mean(all_orr))
        output += "- Mean GMV: {:.2f}\n".format(np.mean(all_gmv))
        output += "- Total iterations: {}\n".format(len(self.buffer))

        return output

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return "RideRewardBuffer(size={}, max_size={})".format(
            len(self.buffer), self.max_size)
