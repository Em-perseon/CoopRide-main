import os
import sys
import hashlib
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
LLM_HDRL_PATH = os.path.abspath(os.path.join(ROOT, "..", "LLM-A-HDRL"))
if os.path.isdir(LLM_HDRL_PATH):
    sys.path.append(LLM_HDRL_PATH)

LLMClient = None
SemanticEncoder = None
try:
    from llm_interface.llm_client import LLMClient  # type: ignore
    from networks.semantic_encoder import SemanticEncoder  # type: ignore
except Exception:
    LLMClient = None
    SemanticEncoder = None


class GlobalInstructionGenerator:
    def __init__(
        self,
        enabled: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
        embedding_dim: int = 64,
        update_steps: int = 30,
        override_text: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.embedding_dim = int(embedding_dim)
        self.update_steps = max(1, int(update_steps))
        self.override_text = override_text
        self.prompt_template = prompt_template

        self.llm_client = None
        if self.enabled and LLMClient is not None:
            try:
                self.llm_client = LLMClient(llm_config or {})
            except Exception:
                self.llm_client = None

        self.encoder = None
        if SemanticEncoder is not None:
            try:
                self.encoder = SemanticEncoder({
                    "embedding_dim": self.embedding_dim,
                    "cache_enabled": True,
                })
                self.encoder.initialize()
                self.embedding_dim = int(self.encoder.embedding_dim)
            except Exception:
                self.encoder = None

        self._last_text = None
        self._last_vector = None

    @classmethod
    def from_env(cls) -> "GlobalInstructionGenerator":
        enabled = os.environ.get("COOPRIDE_LLM_ENABLED", "0") == "1"
        update_steps = int(os.environ.get("COOPRIDE_LLM_UPDATE_STEPS", "30"))
        embedding_dim = int(os.environ.get("COOPRIDE_LLM_EMBED_DIM", "64"))
        override_text = os.environ.get("COOPRIDE_LLM_INSTRUCTION", None)
        prompt_file = os.environ.get("COOPRIDE_LLM_PROMPT_FILE", None)
        prompt_template = None
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

        llm_config = {
            "provider": os.environ.get("COOPRIDE_LLM_PROVIDER", "openai"),
            "model": os.environ.get("COOPRIDE_LLM_MODEL", "gpt-4o-mini"),
            "api_key": os.environ.get("COOPRIDE_LLM_API_KEY"),
            "base_url": os.environ.get("COOPRIDE_LLM_BASE_URL"),
            "temperature": float(os.environ.get("COOPRIDE_LLM_TEMPERATURE", "0.2")),
            "max_tokens": int(os.environ.get("COOPRIDE_LLM_MAX_TOKENS", "128")),
        }

        return cls(
            enabled=enabled,
            llm_config=llm_config,
            embedding_dim=embedding_dim,
            update_steps=update_steps,
            override_text=override_text,
            prompt_template=prompt_template,
        )

    def _summarize_env(self, env) -> Tuple[Dict[str, Any], List[Tuple[int, float]]]:
        total_orders = 0
        total_idle = 0
        gaps = []
        for node in getattr(env, "nodes", []):
            if node is None:
                continue
            orders = float(getattr(node, "real_order_num", 0))
            idle = float(getattr(node, "idle_driver_num", 0))
            gap = orders - idle
            total_orders += orders
            total_idle += idle
            gaps.append((int(getattr(node, "_index", 0)), gap))

        ratio = total_idle / max(1.0, total_orders)
        gaps.sort(key=lambda x: x[1], reverse=True)
        hotspots = gaps[:3]
        stats = {
            "total_orders": int(total_orders),
            "total_idle": int(total_idle),
            "supply_demand_ratio": float(ratio),
        }
        return stats, hotspots

    def _default_instruction(self, stats: Dict[str, Any], hotspots: List[Tuple[int, float]]) -> str:
        if hotspots:
            hotspots_str = ", ".join([f"grid {gid} (gap {gap:.1f})" for gid, gap in hotspots])
        else:
            hotspots_str = "no obvious hotspots"
        if stats["supply_demand_ratio"] < 0.8:
            return f"Prioritize serving orders in {hotspots_str}. Supply is insufficient."
        if stats["supply_demand_ratio"] > 1.2:
            return f"Reposition idle drivers toward {hotspots_str}."
        return f"Balance supply and demand; focus on {hotspots_str}."

    def _build_prompt(self, stats: Dict[str, Any], hotspots: List[Tuple[int, float]]) -> str:
        if self.prompt_template:
            return self.prompt_template.format(stats=stats, hotspots=hotspots)
        hotspots_str = ", ".join([f"grid {gid} (gap {gap:.1f})" for gid, gap in hotspots]) or "none"
        return (
            "You are an expert ride-hailing dispatcher. "
            "Given the current system stats, output ONE short global instruction. "
            f"Total orders: {stats['total_orders']}, total idle drivers: {stats['total_idle']}, "
            f"supply/demand ratio: {stats['supply_demand_ratio']:.2f}. "
            f"Hotspots: {hotspots_str}."
        )

    def _encode_text(self, text: str) -> np.ndarray:
        if self.encoder is not None:
            return np.asarray(self.encoder.encode(text), dtype=np.float32).reshape(-1)
        # deterministic hash embedding fallback
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest, dtype=np.float32)
        if vec.size < self.embedding_dim:
            vec = np.pad(vec, (0, self.embedding_dim - vec.size))
        elif vec.size > self.embedding_dim:
            vec = vec[: self.embedding_dim]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def get_instruction_vector(self, env, step: int) -> np.ndarray:
        stats, hotspots = self._summarize_env(env)
        if self.override_text:
            text = self.override_text
        elif self.llm_client is not None:
            prompt = self._build_prompt(stats, hotspots)
            response = self.llm_client.chat([
                {"role": "system", "content": "You are a city-scale ride-hailing dispatching expert."},
                {"role": "user", "content": prompt},
            ])
            text = response.content.strip() if response.content else ""
            if not text:
                text = self._default_instruction(stats, hotspots)
        else:
            text = self._default_instruction(stats, hotspots)

        if text == self._last_text and self._last_vector is not None:
            return self._last_vector

        vec = self._encode_text(text)
        self._last_text = text
        self._last_vector = vec
        return vec
