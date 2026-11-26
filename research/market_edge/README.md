# ¿Se puede ganar al mercado?

Estudio pre-registrado que busca una ventaja estadística frente al mercado en ocho
estrategias documentadas en la literatura académica, más un modelo de machine learning
como control.

**Resultado:** ninguna de las 8 cumple los cuatro criterios fijados de antemano. Esos
criterios son alpha significativo fuera de muestra tras corregir por múltiples pruebas,
Deflated Sharpe > 0,95, coste de equilibrio > 10 pb y robustez a los parámetros.

## Ficheros

| Fichero | Qué hace |
|---|---|
| `HYPOTHESES.md` | Pre-registro: reglas, parámetros, costes y criterios, commiteado antes de descargar datos |
| `data.py` | Descarga y cachea los datos (Kenneth French Data Library y Yahoo Finance) |
| `backtest.py` | Simuladores con costes, deriva de pesos y ejecución a cierre o apertura |
| `strategies.py` | Las 8 hipótesis (H1–H8) y sus rejillas de robustez |
| `stats.py` | Métricas, alpha con errores Newey-West, bootstrap del Sharpe, PSR/DSR y Holm |
| `run.py` | Ejecuta el estudio y escribe `results/summary.json` |
| `build_report.py` | Genera `report/informe.html` a partir de los resultados |
| `test_backtest.py` | Tests de los simuladores y de ausencia de look-ahead |

Se usan funciones de `gs_quant.timeseries` (`moving_average`, `volatility`,
`max_drawdown`), que funcionan en local sin credenciales.

## Reproducirlo

```bash
pip install -e ../..            # gs-quant y sus dependencias
pip install scipy statsmodels scikit-learn
python data.py                  # descarga y cachea en data/
python -m pytest test_backtest.py -c /dev/null -p no:cacheprovider
python run.py                   # aprox. 1 minuto
python build_report.py
```

Estudio estadístico con fines educativos, no asesoramiento financiero.
