"""Session statistics for real-time analysis.

Tracks aggregated metrics for the current process to support
advisories and summaries without querying the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

@dataclass
class SessionStats:
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_energy_wh: float = 0.0
    total_carbon_g: float = 0.0
    total_cost_usd: float = 0.0
    
    # For pattern detection
    input_token_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    models_used: set[str] = field(default_factory=set)

    def update(self, event: dict[str, Any]) -> None:
        self.total_requests += 1
        
        usage = event.get("usage", {}) or {}
        text = usage.get("text", {}) or {}
        in_tok = text.get("input_tokens", 0)
        out_tok = text.get("output_tokens", 0)
        
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_energy_wh += (event.get("estimated_energy_wh") or 0.0)
        self.total_carbon_g += (event.get("estimated_carbon_g") or 0.0)
        self.total_cost_usd += (event.get("estimated_cost_usd") or 0.0)
        
        # Track for advisory
        if in_tok > 0:
            self.input_token_counts[in_tok] += 1
        
        model = event.get("model")
        if model:
            self.models_used.add(model)

    def summary(self) -> dict[str, Any]:
        # If output is 0 but input exists, ratio is effectively infinite (input)
        if self.total_output_tokens == 0:
            ratio = float(self.total_input_tokens) if self.total_input_tokens > 0 else 0.0
        else:
            ratio = self.total_input_tokens / self.total_output_tokens

        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "average_input_output_ratio": ratio
        }

# Global singleton
_session_stats = SessionStats()

def get_session_stats() -> SessionStats:
    return _session_stats

def track_session_event(event: dict[str, Any]) -> None:
    """Update global session stats."""
    _session_stats.update(event)
