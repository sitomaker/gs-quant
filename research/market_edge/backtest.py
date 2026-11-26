"""Portfolio simulators with transaction costs, weight drift and explicit execution timing."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Result:
    """Daily output of a backtest. ``ret``/``bench``/``rf`` are simple daily returns on the same index."""
    name: str
    hypothesis: str
    daily: pd.DataFrame  # columns: ret, bench, rf, exposure, cost, turnover
    oos_start: pd.Timestamp
    benchmark: str
    trades: Optional[pd.Series] = None  # net return of each round-trip trade, indexed by exit date
    params: dict = field(default_factory=dict)

    def period(self, which: str) -> pd.DataFrame:
        if which == 'oos':
            return self.daily.loc[self.oos_start:]
        if which == 'is':
            return self.daily.loc[:self.oos_start - pd.Timedelta(days=1)]
        return self.daily

    def trades_in(self, which: str) -> Optional[pd.Series]:
        if self.trades is None:
            return None
        if which == 'oos':
            return self.trades.loc[self.oos_start:]
        if which == 'is':
            return self.trades.loc[:self.oos_start - pd.Timedelta(days=1)]
        return self.trades


def simulate_close(target: pd.DataFrame, returns: pd.DataFrame, rf: pd.Series, cost_bps: float = 5.0,
                   borrow_spread: float = 0.005) -> pd.DataFrame:
    """Close-to-close simulator.

    ``target`` holds the weights traded *at the close* of its index date (rows of all-NaN mean
    "no trade, let weights drift"). Day ``t`` earns the weights held after the close of ``t-1``.
    Unallocated capital earns ``rf``; negative cash (leverage) pays ``rf + borrow_spread``.
    Costs are ``cost_bps`` per unit of one-way turnover, charged at the trade close.
    """
    idx = returns.index
    tgt = target.reindex(idx).to_numpy(dtype=float)
    rets = returns.fillna(0.0).to_numpy(dtype=float)
    rfa = rf.reindex(idx).ffill().fillna(0.0).to_numpy(dtype=float)
    n, k = rets.shape
    c = cost_bps / 1e4
    spread = borrow_spread / TRADING_DAYS
    w = np.zeros(k)
    out = np.zeros((n, 4))  # ret, exposure, cost, turnover
    for t in range(n):
        cash = 1.0 - w.sum()
        gross = w @ rets[t] + cash * (rfa[t] if cash >= 0 else rfa[t] + spread)
        out[t, 1] = w.sum()
        w = w * (1.0 + rets[t]) / (1.0 + gross)
        row = tgt[t]
        if not np.isnan(row).all():
            new = np.nan_to_num(row)
            turnover = np.abs(new - w).sum()
            out[t, 2] = turnover * c
            out[t, 3] = turnover
            w = new
        out[t, 0] = (1.0 + gross) * (1.0 - out[t, 2]) - 1.0
    return pd.DataFrame(out, index=idx, columns=['ret', 'exposure', 'cost', 'turnover'])


def simulate_open_close(w_overnight: pd.Series, w_intraday: pd.Series, r_overnight: pd.Series,
                        r_intraday: pd.Series, rf: pd.Series, cost_bps: float = 5.0) -> pd.DataFrame:
    """Single-asset simulator with trading at both the open and the close.

    ``w_overnight[t]``: weight held from the close of ``t-1`` to the open of ``t``.
    ``w_intraday[t]``: weight held from the open of ``t`` to the close of ``t``.
    Trades happen at the open (overnight -> intraday weight) and at the close (intraday -> next overnight).
    """
    idx = r_overnight.index
    won = w_overnight.reindex(idx).fillna(0.0)
    wid = w_intraday.reindex(idx).fillna(0.0)
    rfh = rf.reindex(idx).ffill().fillna(0.0) / 2.0
    c = cost_bps / 1e4
    turn_open = (wid - won).abs()
    turn_close = (won.shift(-1).fillna(won.iloc[-1]) - wid).abs()
    leg_on = 1.0 + won * r_overnight.fillna(0.0) + (1.0 - won) * rfh
    leg_id = 1.0 + wid * r_intraday.fillna(0.0) + (1.0 - wid) * rfh
    ret = leg_on * (1.0 - turn_open * c) * leg_id * (1.0 - turn_close * c) - 1.0
    return pd.DataFrame({'ret': ret, 'exposure': (won + wid) / 2.0, 'cost': (turn_open + turn_close) * c,
                         'turnover': turn_open + turn_close}, index=idx)


def extract_trades(daily: pd.DataFrame, holding: pd.Series) -> pd.Series:
    """Round-trip trades as runs of consecutive days with the same non-null ``holding`` key.

    The trade return compounds the strategy's net daily returns over the run, plus the entry
    day when the entry cost was charged at that day's close (close-executed strategies).
    """
    key = holding.reindex(daily.index)
    ret = daily['ret'].to_numpy()
    cost = daily['cost'].to_numpy()
    k = [None if pd.isna(v) else v for v in key.to_numpy(dtype=object)]
    trades, dates = [], []
    i, n = 0, len(k)
    while i < n:
        if k[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and k[j + 1] == k[i]:
            j += 1
        # entry from flat: the buy cost was charged at the previous close
        start = i - 1 if i > 0 and k[i - 1] is None and cost[i - 1] > 0 else i
        trades.append(np.prod(1.0 + ret[start:j + 1]) - 1.0)
        dates.append(daily.index[j])
        i = j + 1
    return pd.Series(trades, index=pd.DatetimeIndex(dates), dtype=float)
