"""Run every pre-registered hypothesis, its robustness grid and all tests; write results/summary.json."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data
import stats
import strategies as st

OUT = Path(__file__).resolve().parent / 'results'
COST = st.COST_BPS


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not isinstance(v, pd.Series)}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def factors_for(res) -> pd.DataFrame:
    m = data.us_market()
    return m[['Mkt-RF', 'SMB', 'HML', 'MOM']]


def capm(res, which):
    d = res.period(which)
    mkt = (d['bench'] - d['rf']).rename('MKT-RF').to_frame()
    return stats.alpha_regression(d['ret'], d['rf'], mkt)


def evaluate(res, which, boot=True) -> dict:
    d = res.period(which)
    if len(d) < 252:
        return {}
    out = {
        'start': str(d.index[0].date()), 'end': str(d.index[-1].date()),
        'strategy': stats.performance(d['ret'], d['rf']),
        'benchmark': stats.performance(d['bench'], d['rf']),
        'exposure': float(d['exposure'].mean()),
        'turnover_per_year': float(d['turnover'].sum() / (len(d) / 252)),
        'cost_drag_per_year': float(d['cost'].sum() / (len(d) / 252)),
        'periods': stats.period_stats(d['ret'], d['bench']),
        'trades': stats.trade_stats(res.trades_in(which)),
        'capm': capm(res, which),
        'ff4': stats.alpha_regression(d['ret'], d['rf'], factors_for(res)),
    }
    if boot:
        out['sharpe_test'] = stats.sharpe_diff_bootstrap(d['ret'], d['bench'], d['rf'])
    return out


def curves(res) -> dict:
    d = res.daily
    me = st.month_ends(d.index)
    w_s, w_b = (1 + d['ret']).cumprod(), (1 + d['bench']).cumprod()
    dd_s, dd_b = w_s / w_s.cummax() - 1, w_b / w_b.cummax() - 1
    return {'dates': [x.strftime('%Y-%m-%d') for x in me],
            'strategy': [round(float(v), 5) for v in w_s.loc[me]],
            'benchmark': [round(float(v), 5) for v in w_b.loc[me]],
            'dd_strategy': [round(float(v), 4) for v in dd_s.resample('ME').min().reindex(me, method='nearest')],
            'dd_benchmark': [round(float(v), 4) for v in dd_b.resample('ME').min().reindex(me, method='nearest')]}


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    summary, trial_srs, trial_keys, hedged_oos, p_oos = {}, [], [], {}, {}
    for key, (title, fn, primary, grid) in st.HYPOTHESES.items():
        print(f'{key} {title} ...', flush=True)
        prim = fn(**primary)
        prim_free = fn(**primary, cost_bps=0.0)
        ev = {w: evaluate(prim, w) for w in ('full', 'is', 'oos')}
        a5 = ev['oos']['capm']['alpha']
        a0 = capm(prim_free, 'oos')['alpha']
        slope = (a0 - a5) / COST
        breakeven = 0.0 if a0 <= 0 else (math.inf if slope <= 1e-12 else a0 / slope)

        variants = []
        for g in grid:
            r = prim if g == primary else fn(**g)
            c = capm(r, 'oos')
            sr_daily = c['hedged'].mean() / c['hedged'].std()
            trial_srs.append(sr_daily)
            trial_keys.append(key)
            perf = stats.performance(r.period('oos')['ret'], r.period('oos')['rf'])
            variants.append({'name': r.name, 'params': g, 'primary': g == primary, 'alpha_oos': c['alpha'],
                             't_oos': c['t'], 'sharpe_oos': perf['sharpe'], 'max_dd_oos': perf['max_dd']})
        hedged_oos[key] = ev['oos']['capm']['hedged']
        p_oos[key] = ev['oos']['capm']['p_one_sided'] if ev['oos']['capm']['alpha'] > 0 else 1.0
        summary[key] = {
            'title': title, 'name': prim.name, 'benchmark': prim.benchmark, 'params': primary,
            'oos_start': str(prim.oos_start.date()), 'eval': ev, 'breakeven_bps': breakeven,
            'alpha_oos_gross': a0, 'alpha_per_bp': slope,
            'robust_share': float(np.mean([v['alpha_oos'] > 0 for v in variants])),
            'variants': variants, 'curves': curves(prim),
        }
        print(f'   OOS alpha {a5:+.2%}  t={ev["oos"]["capm"]["t"]:.2f}  SR {ev["oos"]["strategy"]["sharpe"]:.2f}'
              f' vs {ev["oos"]["benchmark"]["sharpe"]:.2f}  ({time.time() - t0:.0f}s)', flush=True)

    # multiple testing and overfitting corrections
    holm = stats.holm(p_oos)
    sr_star = stats.expected_max_sharpe(np.array(trial_srs))
    # sensitivity (not pre-registered): H6's cost-driven Sharpe inflates the cross-trial variance
    sr_star_ex6 = stats.expected_max_sharpe(np.array([v for v, k in zip(trial_srs, trial_keys) if k != 'H6']))
    for key, s in summary.items():
        dsr = stats.probabilistic_sharpe(hedged_oos[key], sr_star)
        h = hedged_oos[key]
        s['appraisal_ratio_oos'] = float(h.mean() / h.std() * np.sqrt(252))
        s['dsr_ex_h6'] = stats.probabilistic_sharpe(h, sr_star_ex6)
        s['psr_zero'] = stats.probabilistic_sharpe(h, 0.0)
        ar = s['appraisal_ratio_oos']
        s['years_for_t2'] = float((2 / ar) ** 2) if ar > 0 else None
        s['years_for_t3'] = float((3 / ar) ** 2) if ar > 0 else None
        c1 = s['eval']['oos']['capm']['alpha'] > 0 and holm[key] < 0.05
        c2 = dsr > 0.95
        c3 = s['breakeven_bps'] > 2 * COST
        c4 = s['robust_share'] >= 0.75
        s['tests'] = {'p_holm': holm[key], 'dsr': dsr, 'criteria': {'alpha_oos': c1, 'dsr': c2, 'costs': c3,
                                                                     'robust': c4}, 'edge': bool(c1 and c2 and c3 and c4)}
    meta = {'n_trials': len(trial_srs), 'sr_star_daily': sr_star, 'sr_star_annual': sr_star * np.sqrt(252),
            'sr_star_annual_ex_h6': sr_star_ex6 * np.sqrt(252),
            'trial_sr_annual': [{'h': k, 'sr': v * np.sqrt(252)} for k, v in zip(trial_keys, trial_srs)],
            'cost_bps': COST, 'generated': pd.Timestamp.now().isoformat(timespec='seconds')}

    # robustness of H1 across markets
    others = {}
    for sym in st.OTHER_MARKETS:
        r = st.h1_other_market(sym)
        d = r.daily
        others[sym] = {'start': str(d.index[0].date()),
                       'strategy': stats.performance(d['ret'], d['rf']),
                       'benchmark': stats.performance(d['bench'], d['rf']),
                       'capm': capm(r, 'full'),
                       'oos_strategy': stats.performance(r.period('oos')['ret'], r.period('oos')['rf']),
                       'oos_benchmark': stats.performance(r.period('oos')['bench'], r.period('oos')['rf']),
                       'oos_capm': capm(r, 'oos')}

    (OUT / 'summary.json').write_text(json.dumps(_clean({'meta': meta, 'hypotheses': summary,
                                                          'other_markets': others}), indent=1, ensure_ascii=False))
    print_table(summary, meta, others)
    print(f'done in {time.time() - t0:.0f}s')


def print_table(summary, meta, others):
    print(f'\nTrials: {meta["n_trials"]}  E[max SR] under no skill: {meta["sr_star_annual"]:.2f} (annual, hedged)')
    hdr = f'{"":4}{"estrategia":32}{"OOS desde":>11}{"alpha":>8}{"t":>6}{"p_holm":>8}{"DSR":>6}{"BE pb":>7}' \
          f'{"robust":>7}{"SR":>6}{"SR bm":>6}{"MDD":>7}{"MDD bm":>7}{"win%":>6}  edge'
    print(hdr)
    for k, s in summary.items():
        o = s['eval']['oos']
        tr = o['trades'].get('win_rate')
        print(f'{k:4}{s["name"][:31]:32}{s["oos_start"]:>11}{o["capm"]["alpha"]:>8.2%}{o["capm"]["t"]:>6.2f}'
              f'{s["tests"]["p_holm"]:>8.3f}{s["tests"]["dsr"]:>6.2f}{min(s["breakeven_bps"], 999):>7.1f}'
              f'{s["robust_share"]:>7.0%}{o["strategy"]["sharpe"]:>6.2f}{o["benchmark"]["sharpe"]:>6.2f}'
              f'{o["strategy"]["max_dd"]:>7.0%}{o["benchmark"]["max_dd"]:>7.0%}'
              f'{(tr if tr is not None else float("nan")):>6.0%}  {"SI" if s["tests"]["edge"] else "no"}')
    print('\nH1 en otros mercados (desde inicio): SR estrategia vs B&H, MDD')
    for sym, o in others.items():
        print(f'  {sym:10}{o["start"]:>12}  SR {o["strategy"]["sharpe"]:.2f} vs {o["benchmark"]["sharpe"]:.2f}'
              f'  MDD {o["strategy"]["max_dd"]:.0%} vs {o["benchmark"]["max_dd"]:.0%}'
              f'  alpha {o["capm"]["alpha"]:+.2%} (t={o["capm"]["t"]:.2f})'
              f'  | desde 2007 SR {o["oos_strategy"]["sharpe"]:.2f} vs {o["oos_benchmark"]["sharpe"]:.2f}')


if __name__ == '__main__':
    main()
