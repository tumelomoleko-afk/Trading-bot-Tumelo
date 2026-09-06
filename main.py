"""
SMC / ICT Strategy Engine
-------------------------
Paper-trading / backtesting only.

The engine produces a setup-quality score.
A score of 95/100 does NOT mean a 95% probability of profit.

The engine looks for:
- Liquidity sweeps
- Market structure
- BOS / CHOCH proxy
- Fair Value Gaps
- Displacement
- Order blocks
- Higher-timeframe bias
- Trading session
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class FVG:
    direction: str
    high: float
    low: float
    index: int


@dataclass
class Signal:
    direction: str
    score: int

    entry: float
    stop_loss: float
    take_profit: float

    risk_distance: float
    reward_distance: float
    risk_reward: float

    timestamp: datetime

    flags: Dict[str, bool]
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    "swing_window": 3,
    "liquidity_lookback": 40,

    "atr_period": 14,

    "minimum_score": 90,

    "risk_reward": 2.0,

    "displacement_atr_multiplier": 1.25,

    # UTC session
    "session_start": 7,
    "session_end": 20,
}


# ============================================================
# BASIC CANDLE FUNCTIONS
# ============================================================

def candle_body(candle: Candle) -> float:
    return abs(candle.close - candle.open)


def candle_range(candle: Candle) -> float:
    return candle.high - candle.low


def bullish(candle: Candle) -> bool:
    return candle.close > candle.open


def bearish(candle: Candle) -> bool:
    return candle.close < candle.open


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    candles: List[Candle],
    period: int = 14
) -> Optional[float]:

    if len(candles) < period + 1:
        return None

    true_ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close)
        )

        true_ranges.append(tr)

    recent = true_ranges[-period:]

    if not recent:
        return None

    return sum(recent) / len(recent)


# ============================================================
# EMA
# ============================================================

def ema(values: List[float], period: int) -> Optional[float]:

    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    current_ema = sum(values[:period]) / period

    for value in values[period:]:
        current_ema = (
            (value - current_ema) * multiplier
        ) + current_ema

    return current_ema


# ============================================================
# SWING DETECTION
# ============================================================

def is_swing_high(
    candles: List[Candle],
    index: int,
    window: int
) -> bool:

    if index - window < 0:
        return False

    if index + window >= len(candles):
        return False

    current_high = candles[index].high

    for i in range(
        index - window,
        index + window + 1
    ):
        if i == index:
            continue

        if candles[i].high >= current_high:
            return False

    return True


def is_swing_low(
    candles: List[Candle],
    index: int,
    window: int
) -> bool:

    if index - window < 0:
        return False

    if index + window >= len(candles):
        return False

    current_low = candles[index].low

    for i in range(
        index - window,
        index + window + 1
    ):
        if i == index:
            continue

        if candles[i].low <= current_low:
            return False

    return True


def recent_swing_levels(
    candles: List[Candle],
    current_index: int,
    window: int,
    lookback: int
):

    highs = []
    lows = []

    start = max(
        window,
        current_index - lookback
    )

    # Only use CONFIRMED swings.
    # This prevents future candles from leaking
    # into the current signal.

    end = current_index - window

    if end <= start:
        return highs, lows

    for i in range(start, end):

        if is_swing_high(
            candles,
            i,
            window
        ):
            highs.append(
                candles[i].high
            )

        if is_swing_low(
            candles,
            i,
            window
        ):
            lows.append(
                candles[i].low
            )

    return highs, lows


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(
    candles: List[Candle],
    index: int,
    config: dict
):

    if index <= 0:
        return None

    window = config["swing_window"]
    lookback = config["liquidity_lookback"]

    highs, lows = recent_swing_levels(
        candles,
        index,
        window,
        lookback
    )

    if not highs and not lows:
        return None

    current = candles[index]

    # Bearish liquidity sweep:
    # Price takes previous highs then closes back below.
    if highs:

        previous_high = max(highs)

        if (
            current.high > previous_high
            and current.close < previous_high
        ):
            return "bearish"

    # Bullish liquidity sweep:
    # Price takes previous lows then closes back above.
    if lows:

        previous_low = min(lows)

        if (
            current.low < previous_low
            and current.close > previous_low
        ):
            return "bullish"

    return None


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(
    candles: List[Candle],
    index: int,
