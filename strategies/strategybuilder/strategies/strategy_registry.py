"""
Strategy Registry

Central registry for all available strategies.
New strategies can be registered here.
"""

from typing import Dict, Type, Any
from .mvg_avg_crossover_original import MovingAverageCrossoverOriginal


# Strategy registry
STRATEGIES: Dict[str, Type] = {
    'mvg_avg_crossover_original': MovingAverageCrossoverOriginal,
    'moving_average_crossover_original': MovingAverageCrossoverOriginal,  # Alias
}


def get_strategy(strategy_name: str, params: Dict[str, Any] = None):
    """
    Get a strategy instance by name
    
    Args:
        strategy_name: Name of the strategy
        params: Strategy parameters
    
    Returns:
        Strategy instance
    """
    if strategy_name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(
            f"Strategy '{strategy_name}' not found. "
            f"Available strategies: {available}"
        )
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(params=params)


def list_strategies() -> list:
    """List all available strategies"""
    return list(STRATEGIES.keys())

