"""Build report/informe.html from results/summary.json and report/template.html."""

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {'^GSPC': 'S&P 500 (EE.UU.)', '^N225': 'Nikkei 225 (Japón)', '^FTSE': 'FTSE 100 (R. Unido)',
          '^GDAXI': 'DAX (Alemania)', '^HSI': 'Hang Seng (Hong Kong)', '^IBEX': 'IBEX 35 (España)',
          '^STOXX50E': 'Euro Stoxx 50', 'BTC-USD': 'Bitcoin'}


def r(x, n=4):
    return None if x is None else round(x, n)


def quarterly(c):
    """Downsample monthly curves to quarter ends; drawdowns keep the quarter's worst value."""
    out = {'d': [], 's': [], 'b': [], 'ds': [], 'db': []}
    n = len(c['dates'])
    for i in range(0, n, 3):
        j = min(i + 2, n - 1)
        out['d'].append(c['dates'][j][:7])
        out['s'].append(float(f'{c["strategy"][j]:.4g}'))
        out['b'].append(float(f'{c["benchmark"][j]:.4g}'))
        out['ds'].append(r(min(c['dd_strategy'][i:j + 1]), 3))
        out['db'].append(r(min(c['dd_benchmark'][i:j + 1]), 3))
    return out


def block(e):
    if not e:
        return None
    s, b = e['strategy'], e['benchmark']
    out = {'start': e['start'], 'end': e['end'], 'years': r(s['years'], 1),
           'alpha': r(e['capm']['alpha']), 't': r(e['capm']['t'], 2), 'beta': r(e['capm']['betas']['MKT-RF'], 2),
           'ff4_alpha': r(e['ff4']['alpha']), 'ff4_t': r(e['ff4']['t'], 2),
           'sr': r(s['sharpe'], 2), 'sr_b': r(b['sharpe'], 2), 'cagr': r(s['cagr']), 'cagr_b': r(b['cagr']),
           'vol': r(s['vol']), 'vol_b': r(b['vol']), 'mdd': r(s['max_dd'], 3), 'mdd_b': r(b['max_dd'], 3),
           'exposure': r(e['exposure'], 2), 'cost_drag': r(e['cost_drag_per_year']),
           'pct_years_beat': r(e['periods']['pct_years_beat'], 3), 'pct_years_up': r(e['periods']['pct_years_up'], 3),
           'pct_months_up': r(e['periods']['pct_months_up'], 3), 'n_years': e['periods']['n_years'], 'worst_year': r(e['periods']['worst_year'], 3),
           'worst_year_b': r(e['periods']['worst_year_bench'], 3)}
    if e.get('sharpe_test'):
        out['sr_diff_p'] = r(e['sharpe_test']['p_one_sided'], 3)
    t = e.get('trades') or {}
    if t:
        out.update({'n_trades': t['n_trades'], 'win_rate': r(t['win_rate'], 3), 'avg_win': r(t['avg_win']),
                    'avg_loss': r(t['avg_loss']), 'profit_factor': r(t['profit_factor'], 2),
                    'max_losses': t['max_consecutive_losses']})
    return out


def main():
    s = json.loads((HERE / 'results' / 'summary.json').read_text())
    commit = subprocess.run(['git', 'log', '--format=%h', '-1', '--', 'HYPOTHESES.md'], cwd=HERE,
                            capture_output=True, text=True).stdout.strip()
    hyps = []
    for k, h in s['hypotheses'].items():
        hyps.append({
            'key': k, 'title': h['title'], 'name': h['name'], 'bench': h['benchmark'], 'oos_start': h['oos_start'],
            'full': block(h['eval']['full']), 'is': block(h['eval'].get('is')), 'oos': block(h['eval']['oos']),
            'tests': {'p_holm': r(h['tests']['p_holm'], 3), 'dsr': r(h['tests']['dsr'], 3),
                      'dsr_ex6': r(h['dsr_ex_h6'], 3), 'psr0': r(h['psr_zero'], 3),
                      'breakeven': r(min(h['breakeven_bps'] or 0, 999), 1), 'robust': r(h['robust_share'], 2),
                      'criteria': h['tests']['criteria'], 'edge': h['tests']['edge']},
            'gross_alpha': r(h['alpha_oos_gross']),
            'power': {'ar': r(h['appraisal_ratio_oos'], 3), 't2': r(h['years_for_t2'], 1),
                      't3': r(h['years_for_t3'], 1)},
            'curve': quarterly(h['curves']),
        })
    markets = [{'sym': k, 'label': LABELS[k], 'start': o['start'], 'sr': r(o['strategy']['sharpe'], 2),
                'sr_b': r(o['benchmark']['sharpe'], 2), 'mdd': r(o['strategy']['max_dd'], 3),
                'mdd_b': r(o['benchmark']['max_dd'], 3), 'alpha': r(o['capm']['alpha']), 't': r(o['capm']['t'], 2),
                'oos_sr': r(o['oos_strategy']['sharpe'], 2), 'oos_sr_b': r(o['oos_benchmark']['sharpe'], 2)}
               for k, o in s['other_markets'].items()]
    meta = {'n_trials': s['meta']['n_trials'], 'sr_star': r(s['meta']['sr_star_annual'], 2),
            'sr_star_ex6': r(s['meta']['sr_star_annual_ex_h6'], 2), 'cost_bps': s['meta']['cost_bps'],
            'generated': s['meta']['generated'][:10], 'prereg_commit': commit}
    payload = json.dumps({'meta': meta, 'hyps': hyps, 'markets': markets}, ensure_ascii=False, separators=(',', ':'))
    template = (HERE / 'report' / 'template.html').read_text()
    html = template.replace('/*__DATA__*/null', payload)
    (HERE / 'report' / 'informe.html').write_text(html)
    print(f'payload {len(payload) / 1024:.0f} KB -> report/informe.html ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
