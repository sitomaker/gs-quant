"""Performance metrics and the statistical tests used to decide whether an edge is real."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from backtest import TRADING_DAYS
from gs_quant.timeseries import max_drawdown

EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------- descriptive metrics

def performance(ret: pd.Series, rf: pd.Series) -> dict:
    ex = ret - rf
    years = len(ret) / TRADING_DAYS
    wealth = (1 + ret).cumprod()
    downside = ex[ex < 0].std() * np.sqrt(TRADING_DAYS)
    mdd = float(max_drawdown(wealth).min())  # gs_quant, fraction (negative)
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    return {
        'cagr': cagr,
        'vol': ret.std() * np.sqrt(TRADING_DAYS),
        'sharpe': ex.mean() / ex.std() * np.sqrt(TRADING_DAYS),
        'sortino': ex.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        'max_dd': mdd,
        'calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
        'years': years,
    }


def period_stats(ret: pd.Series, bench: pd.Series) -> dict:
    """How often the strategy wins: months/years with gains and months/years beating the benchmark."""
    g = pd.concat({'s': ret, 'b': bench}, axis=1)
    mon = (1 + g).groupby([g.index.year, g.index.month]).prod() - 1
    yr = (1 + g).groupby(g.index.year).prod() - 1
    yr = yr[(g.groupby(g.index.year).size() > 200).values]  # complete years only
    return {
        'pct_months_up': float((mon['s'] > 0).mean()),
        'pct_months_beat': float((mon['s'] > mon['b']).mean()),
        'pct_years_up': float((yr['s'] > 0).mean()),
        'pct_years_beat': float((yr['s'] > yr['b']).mean()),
        'worst_year': float(yr['s'].min()),
        'worst_year_bench': float(yr['b'].min()),
        'n_years': int(len(yr)),
    }


def trade_stats(trades: pd.Series) -> dict:
    if trades is None or len(trades) == 0:
        return {}
    wins, losses = trades[trades > 0], trades[trades <= 0]
    n = len(trades)
    win_rate = len(wins) / n
    return {
        'n_trades': n,
        'win_rate': win_rate,
        'avg_win': float(wins.mean()) if len(wins) else 0.0,
        'avg_loss': float(losses.mean()) if len(losses) else 0.0,
        'expectancy': float(trades.mean()),
        'profit_factor': float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf,
        'max_consecutive_losses': int(_max_run(trades.to_numpy() <= 0)),
        # binomial test: is the win rate above 50 %?
        'p_win_rate_gt_50': float(sps.binomtest(len(wins), n, 0.5, alternative='greater').pvalue),
    }


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ----------------------------------------------------------------------------- inference

def _hac_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def alpha_regression(ret: pd.Series, rf: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of excess returns on factors with Newey-West standard errors.

    Returns the annualised alpha, its t-stat and one-sided p-value (H1: alpha > 0), betas and
    the beta-hedged return series (alpha + residual) used for the Deflated Sharpe Ratio.
    """
    df = pd.concat([(ret - rf).rename('y'), factors], axis=1, join='inner').dropna()
    X = sm.add_constant(df[factors.columns])
    fit = sm.OLS(df['y'], X).fit(cov_type='HAC', cov_kwds={'maxlags': _hac_lags(len(df))})
    t = float(fit.tvalues['const'])
    hedged = df['y'] - df[factors.columns] @ fit.params[factors.columns]
    return {
        'alpha': float(fit.params['const'] * TRADING_DAYS),
        't': t,
        'p_one_sided': float(1 - sps.norm.cdf(t)),
        'betas': {k: float(v) for k, v in fit.params[factors.columns].items()},
        'hedged': hedged,
        'n': int(len(df)),
    }


def sharpe_diff_bootstrap(ret: pd.Series, bench: pd.Series, rf: pd.Series, n_boot: int = 2000,
                          block: int = 20, seed: int = 0) -> dict:
    """Stationary bootstrap (Politis & Romano 1994) test of H1: Sharpe(strategy) > Sharpe(benchmark).

    Paired resampling keeps the cross-correlation between both series; blocks keep volatility clustering.
    """
    x = (ret - rf).to_numpy()
    y = (bench - rf).to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    p_new = 1.0 / block
    idx = rng.integers(0, n, n_boot)
    sx, sxx, sy, syy = x[idx].copy(), x[idx] ** 2, y[idx].copy(), y[idx] ** 2
    for _ in range(1, n):
        jump = rng.random(n_boot) < p_new
        idx = np.where(jump, rng.integers(0, n, n_boot), (idx + 1) % n)
        xi, yi = x[idx], y[idx]
        sx += xi
        sxx += xi * xi
        sy += yi
        syy += yi * yi
    sr_x = (sx / n) / np.sqrt(sxx / n - (sx / n) ** 2)
    sr_y = (sy / n) / np.sqrt(syy / n - (sy / n) ** 2)
    boot = (sr_x - sr_y) * np.sqrt(TRADING_DAYS)
    obs = (x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(TRADING_DAYS)
    p = float(np.mean(boot - obs >= obs))  # centred under H0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'diff': float(obs), 'ci_low': float(lo), 'ci_high': float(hi), 'p_one_sided': p}


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float) -> float:
    """PSR (Bailey & Lopez de Prado 2012) on per-period (non-annualised) Sharpe ratios."""
    r = returns.dropna().to_numpy()
    sr = r.mean() / r.std(ddof=1)
    skew = sps.skew(r)
    kurt = sps.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(len(r) - 1) / denom))


def expected_max_sharpe(trial_srs: np.ndarray) -> float:
    """Expected maximum per-period Sharpe among N unskilled trials (Bailey & Lopez de Prado 2014)."""
    n = len(trial_srs)
    var = np.var(trial_srs, ddof=1)
    return float(np.sqrt(var) * ((1 - EULER_GAMMA) * sps.norm.ppf(1 - 1 / n)
                                 + EULER_GAMMA * sps.norm.ppf(1 - 1 / (n * np.e))))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
