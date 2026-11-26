# Pre-registro de hipótesis

Este documento se escribió **antes de descargar o mirar ningún dato**. Fija las
reglas, parámetros, costes y criterios de éxito de antemano para que el resultado
no dependa de probar variantes hasta que "algo salga bien" (data snooping).
Cualquier cambio posterior a este fichero queda en el historial de git.

## Qué significa "ganar al mercado"

No existe una estrategia que gane en todas las operaciones. Una **ventaja
estadística** es una esperanza matemática positiva tras costes, medida sobre
muchas operaciones y en datos que no se usaron para diseñarla.

Para este estudio, una estrategia "gana al mercado" si su **alpha** (rentabilidad
en exceso ajustada por la exposición al mercado, beta) es positivo y
estadísticamente significativo. Batir la rentabilidad bruta asumiendo más riesgo
no cuenta como ventaja.

## Datos

| Fuente | Serie | Desde |
|---|---|---|
| Kenneth French Data Library | Mercado EE.UU. (CRSP value-weighted, retorno total), RF, SMB, HML, Mom (diario) | 1926 |
| Kenneth French Data Library | 30 carteras de industria value-weighted (diario) | 1926 |
| Yahoo Finance | SPY, EFA, AGG (OHLC ajustado por dividendos y splits) | 1993 / 2001 / 2003 |
| Yahoo Finance | Índices internacionales y BTC (solo robustez, sin dividendos) | varias |

## Supuestos de ejecución (comunes)

- Sin mirar al futuro: una señal calculada con el cierre del día *t* se ejecuta
  al cierre de *t+1*, salvo en H5 y H6, que se ejecutan a la apertura de *t+1*,
  y en H4, cuyo calendario se conoce de antemano.
- Costes: **5 pb por operación en un sentido** sobre el valor negociado. También
  se publica el **coste de equilibrio** (*breakeven*), es decir, el coste al que
  el alpha desaparece.
- La liquidez rinde el tipo libre de riesgo (RF de Kenneth French).
- El apalancamiento (solo en H2) paga RF + 0,5 % anual.
- No se incluyen impuestos.

## Hipótesis (parámetros fijados de antemano)

El periodo **fuera de muestra (OOS)** de cada hipótesis empieza en el año de su
publicación académica. Ese es el test más limpio: ni los autores ni yo pudimos
ver esos datos al diseñar la regla.

| # | Hipótesis | Regla principal | Activo | Publicación → OOS |
|---|---|---|---|---|
| H1 | Seguimiento de tendencia (Siegel 1994; Faber 2007) | Invertido si el índice de retorno total > su media móvil de 200 sesiones; si no, liquidez | Mercado EE.UU. | 2007 → 2007–hoy |
| H2 | Gestión por volatilidad (Moreira & Muir 2017) | Peso = min(2, 16 % / vol. realizada de 21 días), rebalanceo mensual | Mercado EE.UU. | 2017 → 2017–hoy |
| H3 | Momentum de industrias (Moskowitz & Grinblatt 1999) | Cada fin de mes, comprar las 5 de 30 industrias con mayor retorno en 6 meses, a pesos iguales | 30 industrias | 1999 → 1999–hoy |
| H4 | Efecto cambio de mes (Ariel 1987; Lakonishok & Smidt 1988) | Invertido solo del último día hábil del mes al 3.º del siguiente | Mercado EE.UU. | 1988 → 1988–hoy |
| H5 | Reversión a corto plazo, RSI(2) (Connors & Alvarez 2008) | Comprar si cierre > MM200 y RSI(2) < 10; vender cuando cierre > MM5 | SPY | 2008 → 2009–hoy |
| H6 | Efecto overnight (Cliff, Cooper & Gulen 2008) | Comprar al cierre, vender a la apertura, todos los días | SPY | 2008 → 2008–hoy |
| H7 | Dual momentum (Antonacci 2014) | Si el retorno de SPY en 12 meses > RF: SPY o EFA, el que más haya subido; si no, AGG | SPY/EFA/AGG | 2014 → 2014–hoy |
| H8 | Machine learning (control) | Gradient boosting que predice el signo del retorno semanal; reentrenado cada año con ventana creciente | Mercado EE.UU. | walk-forward 1950–hoy |

H8 no tiene respaldo académico de partida. Se incluye como control: es lo que
muchos intentan primero, y comprueba si un modelo flexible encuentra algo que
las reglas simples no encuentran.

## Rejilla de robustez (pre-registrada; no se usa para elegir parámetros)

- H1: media móvil ∈ {50, 100, 150, 200, 250, 300} sesiones
- H2: ventana de volatilidad ∈ {10, 21, 42, 63} × tope de peso ∈ {1,0; 1,5; 2,0}
- H3: formación ∈ {3, 6, 9, 12} meses × top-k ∈ {3, 5, 8}
- H4: ventana ∈ {[-1,+3], [-1,+2], [-2,+3], [-1,+4], [-3,+3]}
- H5: umbral de RSI ∈ {5, 10, 15, 20, 25} × media de salida ∈ {3, 5, 10}
- H6: solo sensibilidad a costes
- H7: retrospectiva ∈ {3, 6, 9, 12} meses
- H8: modelo ∈ {gradient boosting, logística} × horizonte ∈ {semana, mes}

Todas las variantes cuentan como **ensayos** al calcular el Deflated Sharpe Ratio.

## Criterio de "ventaja estadística probada" (todas deben cumplirse)

1. **Alpha OOS significativo**: alpha CAPM > 0 en el periodo
   post-publicación, con errores Newey-West y p-valor ajustado por Holm sobre
   las 8 hipótesis < 0,05.
2. **Sobrevive al sobreajuste**: Deflated Sharpe Ratio (Bailey & López de
   Prado 2014) de la cartera cubierta de beta > 0,95, con N = todos los ensayos.
3. **Sobrevive a los costes**: coste de equilibrio > 10 pb por operación (el
   doble del supuesto).
4. **Es robusta**: alpha > 0 en al menos el 75 % de las variantes de la rejilla.

Además, con carácter informativo: test de diferencia de Sharpe frente a comprar
y mantener (bootstrap estacionario), alpha frente a Fama-French 3 factores +
momentum, tasa de acierto por operación, años y ventanas móviles en que la
estrategia bate al mercado, y consistencia en otros mercados (H1).
