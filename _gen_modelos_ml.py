import json

def md(s):  return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = []

cells.append(md(
"""# Modelos de Machine Learning — Walk-Forward + SHAP

8 modelos (4 puntuales + 4 por cuantiles) sobre NVDA, MSFT y GOOGL, evaluados con **walk-forward de ventana expandible** y el **mismo motor de backtest** que `metodos_tradicionales.ipynb`.

- **Selección de variables:** VIF iterativo + ADF (set fijo de 9). SHAP NO selecciona.
- **Folds:** test = 1 año; 9 folds 2017→2025; train expandible anclado en 2011; gap de 1 día (anti-fuga).
- **Señal puntual:** compra `r̂ > θ`, liquida `r̂ < −θ`. **Señal cuantil:** compra `Q0.1 > θ`, liquida `Q0.9 < −θ`.
- **OOS agrupado** (manzana con manzana vs tradicionales) + **Sharpe por fold** (estabilidad entre regímenes).
- **SHAP** al final, solo interpretación (Objetivo 1).

Detalle en `metodología/plan_walkforward_shap.md`."""))

cells.append(md("## 0. Instalación de dependencias (ejecutar una vez)"))
cells.append(code(
"""import sys
# tensorflow es pesado: si no vas a correr los LSTM, puedes quitarlo de la lista.
!{sys.executable} -m pip install -q numpy pandas scikit-learn xgboost shap matplotlib statsmodels quantile-forest tensorflow"""))

cells.append(md("## 1. Imports y detección de paquetes disponibles"))
cells.append(code(
"""import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)

# Los modelos opcionales se registran solo si su paquete esta disponible.
DISPONIBLES = {}
try:
    from xgboost import XGBRegressor; DISPONIBLES['xgboost'] = True
except Exception as e:
    DISPONIBLES['xgboost'] = False; print('xgboost no disponible:', e)
try:
    from quantile_forest import RandomForestQuantileRegressor; DISPONIBLES['qforest'] = True
except Exception as e:
    DISPONIBLES['qforest'] = False; print('quantile-forest no disponible:', e)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense
    tf.random.set_seed(0); DISPONIBLES['tensorflow'] = True
except Exception as e:
    DISPONIBLES['tensorflow'] = False; print('tensorflow no disponible:', e)
try:
    import shap; DISPONIBLES['shap'] = True
except Exception as e:
    DISPONIBLES['shap'] = False; print('shap no disponible:', e)

print('Disponibles:', DISPONIBLES)"""))

cells.append(md("## 2. Configuración (idéntica a los tradicionales donde aplica)"))
cells.append(code(
"""CAPITAL_INICIAL = 10_000
COSTO = 0.001          # 0.1% por operacion
DIAS_ANIO = 252
THETA = 0.002          # umbral de la senal (0.2%)
GAP = 1                # hueco train->test (el target usa shift(-1))
L_LSTM = 20            # memoria del LSTM (dias)

TICKERS = ['NVDA', 'MSFT', 'GOOGL']
DATA_DIR = Path('datos')

# Set curado por VIF + ADF (9 variables, todas estacionarias)
FEATURES = ['BB_upper_diff', 'Gold_ret', 'MACD_signal', 'Momentum_10',
            'NASDAQ_ret', 'RSI_14', 'SP500_ret', 'Volume', 'WilliamsR_14']

# Walk-forward de ventana expandible
ANIO_INICIO_OOS = 2017
ANIO_FIN_OOS = 2025
QUANTILES = [0.1, 0.5, 0.9]"""))

cells.append(md("## 3. Carga de datos y feature engineering"))
cells.append(code(
"""def cargar(ticker):
    df = pd.read_csv(DATA_DIR / f'{ticker}_dataset.csv', parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    # Transformacion de estacionariedad decidida por ADF: BB_upper -> diferencia lag-1
    df['BB_upper_diff'] = df['BB_upper'].diff()
    # La ultima fila no tiene dia siguiente real (target NaN) -> fuera
    df = df[df['Target_Retorno_1d'].notna()]
    # Warm-up: descarta filas iniciales con NaN en alguna feature curada
    df = df.dropna(subset=FEATURES)
    return df"""))

cells.append(md("## 4. Folds del walk-forward (ventana expandible + gap anti-fuga)"))
cells.append(code(
"""def generar_folds(idx):
    # Para cada anio de test, train = todo lo anterior menos GAP filas (hueco anti-fuga).
    folds = []
    for anio in range(ANIO_INICIO_OOS, ANIO_FIN_OOS + 1):
        test_pos = np.where(idx.year == anio)[0]
        if len(test_pos) == 0:
            continue
        train_pos = np.arange(0, test_pos[0])
        if GAP > 0:
            train_pos = train_pos[:-GAP]
        if len(train_pos) == 0:
            continue
        folds.append((anio, train_pos, test_pos))
    return folds"""))

cells.append(md("## 5. Motor de backtest (IDÉNTICO a `metodos_tradicionales.ipynb`)"))
cells.append(code(
"""# Mismo motor: garantiza que ML y tradicionales se midan con la misma vara.
def _maquina_estado(compra, venta):
    c = compra.to_numpy(); v = venta.to_numpy()
    pos = np.zeros(len(c), dtype=int); estado = 0
    for i in range(len(c)):
        if estado == 0 and c[i]:
            estado = 1
        elif estado == 1 and v[i]:
            estado = 0
        pos[i] = estado
    return pd.Series(pos, index=compra.index)

def backtest(pos, ret, capital=CAPITAL_INICIAL, costo=COSTO):
    pos = pos.astype(float)
    cambios = pos.diff().abs(); cambios.iloc[0] = abs(pos.iloc[0])
    r = pos * ret - costo * cambios
    r.iloc[-1] = r.iloc[-1] - costo * pos.iloc[-1]
    equity = capital * (1.0 + r).cumprod()
    return r, equity

def metricas(r, equity, pos, ret):
    n = len(r); cap_final = equity.iloc[-1]
    ret_total = cap_final / CAPITAL_INICIAL - 1.0
    ret_anual = (cap_final / CAPITAL_INICIAL) ** (DIAS_ANIO / n) - 1.0
    sigma = r.std(ddof=1)
    sharpe = np.sqrt(DIAS_ANIO) * r.mean() / sigma if sigma > 0 else np.nan
    vol = sigma * np.sqrt(DIAS_ANIO)
    max_dd = (equity / equity.cummax() - 1.0).min()
    en_mercado = pos == 1
    hit = (ret[en_mercado] > 0).mean() if en_mercado.any() else np.nan
    n_compras = int((pos.diff() > 0).sum() + (pos.iloc[0] > 0))
    return {'Retorno Total': ret_total, 'Retorno Anual': ret_anual, 'Sharpe': sharpe,
            'Volatilidad': vol, 'Max Drawdown': max_dd, 'Hit Ratio': hit,
            '% en mercado': en_mercado.mean(), 'N compras': n_compras}

# Senales tradicionales (para re-reportarlas sobre el MISMO tramo OOS)
def senal_buy_hold(df): return pd.Series(1, index=df.index)
def senal_sma(df): return (df['SMA_20'] > df['SMA_50']).astype(int)
def senal_macd(df): return (df['MACD'] > df['MACD_signal']).astype(int)
def senal_rsi(df):
    rsi = df['RSI_14']; prev = rsi.shift(1)
    compra = (prev < 30) & (rsi >= 30); venta = (prev > 70) & (rsi <= 70)
    return _maquina_estado(compra.fillna(False), venta.fillna(False))
def senal_mean_reversion(df):
    compra = df['Close'] <= df['BB_lower']; venta = df['Close'] >= df['BB_upper']
    return _maquina_estado(compra, venta)
SENALES_TRAD = {'Buy & Hold': senal_buy_hold, 'SMA Crossover': senal_sma, 'MACD': senal_macd,
                'RSI': senal_rsi, 'Mean Reversion': senal_mean_reversion}"""))

cells.append(md("## 6. Señales de trading desde las predicciones ML"))
cells.append(code(
"""def senal_puntual(pred):
    # Histeresis: compra si r_hat > theta, liquida si r_hat < -theta.
    return _maquina_estado(pred > THETA, pred < -THETA)

def senal_cuantil(q50):
    # Gatillo por la MEDIANA con histeresis (igual que el puntual): compra Q0.5 > theta, liquida Q0.5 < -theta.
    # El intervalo [Q0.1, Q0.9] NO se usa para operar: se reserva para medir incertidumbre (Objetivo 2).
    return _maquina_estado(q50 > THETA, q50 < -THETA)"""))

cells.append(md(
"""## 7. Registro de modelos

Cada modelo expone `tipo` ('puntual'/'cuantil'), `escalado` (None/'standard'), `secuencial` (bool) y
`fit/predict` (o `fit_predict` para los secuenciales). El escalado por fold lo aplica el bucle."""))
cells.append(code(
"""# --- Puntuales ---
class MLineal:
    tipo='puntual'; escalado='standard'; secuencial=False
    def fit(self, X, y): self.m = LinearRegression().fit(X, y)
    def predict(self, X): return self.m.predict(X)

class MRandomForest:
    tipo='puntual'; escalado=None; secuencial=False
    def fit(self, X, y): self.m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X, y)
    def predict(self, X): return self.m.predict(X)

class MXGBoost:
    tipo='puntual'; escalado=None; secuencial=False
    def fit(self, X, y):
        self.m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=0, tree_method='hist').fit(X, y)
    def predict(self, X): return self.m.predict(X)

# --- Cuantiles (devuelven matriz n x 3: q10, q50, q90) ---
class MCuantLineal:
    tipo='cuantil'; escalado='standard'; secuencial=False
    def fit(self, X, y):
        self.ms = [QuantileRegressor(quantile=q, alpha=0.0, solver='highs').fit(X, y) for q in QUANTILES]
    def predict(self, X): return np.column_stack([m.predict(X) for m in self.ms])

class MCuantRF:
    tipo='cuantil'; escalado=None; secuencial=False
    def fit(self, X, y):
        self.m = RandomForestQuantileRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X, y)
    def predict(self, X): return np.asarray(self.m.predict(X, quantiles=QUANTILES))

class MCuantXGB:
    # XGBoost nativo multi-cuantil (reg:quantileerror, requiere xgboost>=2.0).
    # TODO tesina: para fidelidad con el Cap. 5, sustituir por la pinball-arctan de Sluijterman (2025).
    tipo='cuantil'; escalado=None; secuencial=False
    def fit(self, X, y):
        self.m = XGBRegressor(objective='reg:quantileerror', quantile_alpha=np.array(QUANTILES),
                              n_estimators=400, max_depth=4, learning_rate=0.05,
                              tree_method='hist', random_state=0).fit(X, y)
    def predict(self, X):
        p = np.asarray(self.m.predict(X))
        return p if p.ndim == 2 else p.reshape(-1, len(QUANTILES))"""))

cells.append(code(
"""# --- Secuenciales (LSTM): hacen su propio escalado MinMax y construccion de secuencias ---
def _make_seq(arr, L):
    if len(arr) < L:
        return np.empty((0, L, arr.shape[1]))
    return np.stack([arr[i-L+1:i+1] for i in range(L-1, len(arr))])

class MLSTM:
    tipo='puntual'; escalado=None; secuencial=True
    def fit_predict(self, Xtr, ytr, Xte):
        sc = MinMaxScaler().fit(Xtr.values)
        Xtr_s, Xte_s = sc.transform(Xtr.values), sc.transform(Xte.values)
        Xseq = _make_seq(Xtr_s, L_LSTM); yseq = ytr.values[L_LSTM-1:]
        Xcat = np.vstack([Xtr_s[-(L_LSTM-1):], Xte_s]); Xseq_te = _make_seq(Xcat, L_LSTM)
        m = Sequential([Input((L_LSTM, Xtr_s.shape[1])), LSTM(32), Dense(1)])
        m.compile(optimizer='adam', loss='mse')
        m.fit(Xseq, yseq, epochs=20, batch_size=32, verbose=0)
        return m.predict(Xseq_te, verbose=0).ravel()

def _pinball(qs):
    qs = tf.constant(qs, dtype=tf.float32)
    def loss(yt, yp):
        e = yt - yp
        return tf.reduce_mean(tf.maximum(qs * e, (qs - 1.0) * e))
    return loss

class MCuantLSTM:
    tipo='cuantil'; escalado=None; secuencial=True
    def fit_predict(self, Xtr, ytr, Xte):
        sc = MinMaxScaler().fit(Xtr.values)
        Xtr_s, Xte_s = sc.transform(Xtr.values), sc.transform(Xte.values)
        Xseq = _make_seq(Xtr_s, L_LSTM); yseq = ytr.values[L_LSTM-1:].reshape(-1, 1)
        Xcat = np.vstack([Xtr_s[-(L_LSTM-1):], Xte_s]); Xseq_te = _make_seq(Xcat, L_LSTM)
        m = Sequential([Input((L_LSTM, Xtr_s.shape[1])), LSTM(32), Dense(len(QUANTILES))])
        m.compile(optimizer='adam', loss=_pinball(QUANTILES))
        m.fit(Xseq, yseq, epochs=20, batch_size=32, verbose=0)
        return m.predict(Xseq_te, verbose=0)"""))

cells.append(code(
"""def construir_modelos():
    M = {'Regresion Lineal': MLineal(), 'Random Forest': MRandomForest(),
         'Reg. Lineal Cuantil': MCuantLineal()}
    if DISPONIBLES['xgboost']:
        M['XGBoost'] = MXGBoost(); M['XGBoost Cuantil'] = MCuantXGB()
    if DISPONIBLES['qforest']:
        M['Random Forest Cuantil'] = MCuantRF()
    if DISPONIBLES['tensorflow']:
        M['LSTM'] = MLSTM(); M['LSTM Cuantil'] = MCuantLSTM()
    return M

MODELOS = construir_modelos()
print('Modelos a evaluar:', list(MODELOS.keys()))"""))

cells.append(md(
"""## 8. Ejecución del walk-forward (pesada — corre una vez)

Para cada acción y modelo se entrena/predice fold por fold y se **cachean las predicciones** a
`predicciones_oos.csv`. Las reglas de señal y el backtest se aplican en la celda 9 (barata): así,
para probar otra regla **no hay que reentrenar** (basta re-correr la celda 9)."""))
cells.append(code(
"""def predecir_oos(modelo, X, y, folds):
    partes = []
    for anio, tr, te in folds:
        Xtr, ytr, Xte = X.iloc[tr], y.iloc[tr], X.iloc[te]
        idx_te = X.index[te]
        if modelo.secuencial:
            p = modelo.fit_predict(Xtr, ytr, Xte)
        else:
            if modelo.escalado == 'standard':
                sc = StandardScaler().fit(Xtr.values)
                a_tr, a_te = sc.transform(Xtr.values), sc.transform(Xte.values)
            else:
                a_tr, a_te = Xtr.values, Xte.values
            modelo.fit(a_tr, ytr.values); p = modelo.predict(a_te)
        p = np.asarray(p)
        if modelo.tipo == 'puntual':
            df_p = pd.DataFrame({'pred': p.ravel()}, index=idx_te)
        else:
            df_p = pd.DataFrame(p, columns=['q10', 'q50', 'q90'], index=idx_te)
        df_p['fold'] = anio
        partes.append(df_p)
    return pd.concat(partes)


# Corre el walk-forward UNA vez y cachea las predicciones a disco.
registros = []
for ticker in TICKERS:
    df = cargar(ticker)
    X, y = df[FEATURES], df['Target_Retorno_1d']
    folds = generar_folds(df.index)
    for nombre, modelo in MODELOS.items():
        pred = predecir_oos(modelo, X, y, folds)
        out = pd.DataFrame({'Fecha': pred.index})
        out['Accion'] = ticker; out['Metodo'] = nombre; out['Familia'] = modelo.tipo
        out['fold'] = pred['fold'].values
        out['ret'] = y.loc[pred.index].values          # retorno realizado (para el backtest)
        if modelo.tipo == 'puntual':
            out['pred'] = pred['pred'].values
        else:
            out['q10'] = pred['q10'].values
            out['q50'] = pred['q50'].values
            out['q90'] = pred['q90'].values
        registros.append(out)
    print(f'{ticker}: predicciones listas ({len(folds)} folds, {len(MODELOS)} modelos)')

pred_long = pd.concat(registros, ignore_index=True)
pred_long.to_csv('predicciones_oos.csv', index=False, encoding='utf-8')
print('Guardado: predicciones_oos.csv (', len(pred_long), 'filas ).')"""))

cells.append(md(
"""## 9. Evaluación de reglas → resultados (barata, re-ejecutable sin reentrenar)

Aplica las reglas de señal a las predicciones cacheadas y corre el motor. **Para probar otra regla:**
cambia `senal_cuantil`/`senal_puntual` (celda 6) y re-corre SOLO esta celda — no hace falta reentrenar."""))
cells.append(code(
"""def evaluar(pred_long):
    filas, por_fold, curvas = [], [], {}
    for (ticker, nombre), g in pred_long.groupby(['Accion', 'Metodo'], sort=False):
        g = g.sort_values('Fecha')
        idx = pd.DatetimeIndex(pd.to_datetime(g['Fecha'].values))
        familia = g['Familia'].iloc[0]
        r_oos = pd.Series(g['ret'].values, index=idx)
        if familia == 'puntual':
            punto = pd.Series(g['pred'].values, index=idx)
            pos = senal_puntual(punto)
        else:
            punto = pd.Series(g['q50'].values, index=idx)
            pos = senal_cuantil(punto)
        r, equity = backtest(pos, r_oos)
        m = metricas(r, equity, pos, r_oos)
        err = r_oos.values - punto.values
        m['MSE'] = float(np.mean(err ** 2))
        m['RSE'] = float(np.sqrt(np.sum(err ** 2) / (len(err) - len(FEATURES))))
        m['Hit Ratio Pred'] = float(np.mean(np.sign(punto.values) == np.sign(r_oos.values)))
        m.update({'Accion': ticker, 'Metodo': nombre, 'Familia': familia})
        filas.append(m); curvas[(ticker, nombre)] = equity
        folds_g = pd.Series(g['fold'].values, index=idx)
        for anio in sorted(folds_g.unique()):
            rr = r.loc[folds_g.index[folds_g == anio]]
            sg = rr.std(ddof=1)
            por_fold.append({'Accion': ticker, 'Metodo': nombre, 'Fold': int(anio),
                             'Sharpe': np.sqrt(DIAS_ANIO) * rr.mean() / sg if sg > 0 else np.nan})
    return pd.DataFrame(filas), pd.DataFrame(por_fold), curvas

resultados_raw, pf, curvas_ml = evaluar(pred_long)
cols = ['Familia', 'MSE', 'RSE', 'Hit Ratio Pred', 'Retorno Total', 'Retorno Anual', 'Sharpe',
        'Volatilidad', 'Max Drawdown', 'Hit Ratio', '% en mercado', 'N compras']
resultados_ml = resultados_raw.set_index(['Accion', 'Metodo'])[cols]
resultados_ml.to_csv('resultados_ml.csv', encoding='utf-8')
pf.to_csv('resultados_ml_por_fold.csv', index=False, encoding='utf-8')
print('Guardado: resultados_ml.csv, resultados_ml_por_fold.csv')
resultados_ml.round(4)"""))

cells.append(md("## 10. Tradicionales sobre el MISMO tramo OOS (manzana con manzana)"))
cells.append(code(
"""filas_trad, curvas_trad = [], {}
for ticker in TICKERS:
    df = cargar(ticker)
    folds = generar_folds(df.index)
    oos_idx = df.index[np.concatenate([te for _, _, te in folds])]
    ret = df['Target_Retorno_1d']
    for nombre, fn in SENALES_TRAD.items():
        pos = fn(df).loc[oos_idx]          # senal con warm-up de toda la historia, evaluada solo en OOS
        r_oos = ret.loc[oos_idx]
        r, equity = backtest(pos, r_oos)
        m = metricas(r, equity, pos, r_oos)
        m.update({'Accion': ticker, 'Metodo': nombre, 'Familia': 'tradicional'})
        filas_trad.append(m)
        curvas_trad[(ticker, nombre)] = equity

resultados_trad_oos = pd.DataFrame(filas_trad).set_index(['Accion', 'Metodo'])
resultados_trad_oos.to_csv('resultados_tradicionales_oos.csv', encoding='utf-8')

# Tabla combinada ordenada por Sharpe (la metrica principal)
comp = pd.concat([
    resultados_ml.reset_index()[['Accion', 'Metodo', 'Familia', 'Sharpe', 'Retorno Total', 'Max Drawdown']],
    resultados_trad_oos.reset_index()[['Accion', 'Metodo', 'Familia', 'Sharpe', 'Retorno Total', 'Max Drawdown']],
]).set_index(['Accion', 'Metodo'])
comp.sort_values(['Accion', 'Sharpe'], ascending=[True, False]).round(3)"""))

cells.append(md("## 11. Curvas de capital OOS (escala log): ML vs Buy & Hold"))
cells.append(code(
"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, ticker in zip(axes, TICKERS):
    for nombre in MODELOS:
        ax.plot(curvas_ml[(ticker, nombre)], label=nombre)
    ax.plot(curvas_trad[(ticker, 'Buy & Hold')], 'k--', lw=2, label='Buy & Hold (trad)')
    ax.set_yscale('log'); ax.set_title(f'{ticker} — OOS {ANIO_INICIO_OOS}-{ANIO_FIN_OOS}')
    ax.set_xlabel('Fecha'); ax.set_ylabel('Capital ($, log)'); ax.legend(fontsize=7)
fig.tight_layout(); plt.show()"""))

cells.append(md("## 12. Estabilidad entre regímenes (Sharpe por fold)"))
cells.append(code(
"""# 'pf' (Sharpe por fold) ya fue calculado por evaluar() en la celda 9
tabla_estab = pf.pivot_table(index=['Accion', 'Metodo'], columns='Fold', values='Sharpe').round(2)
display(tabla_estab)
print('\\nSharpe entre folds (media +/- desviacion):')
pf.groupby(['Accion', 'Metodo'])['Sharpe'].agg(['mean', 'std']).round(3)"""))

cells.append(md(
"""## 13. Calidad de los intervalos — incertidumbre (Objetivo 2)

Los cuantiles `[Q0.1, Q0.9]` ya no operan; aquí se evalúan como tales: la **cobertura empírica** debería
rondar **0.80** (un intervalo 10-90 bien calibrado contiene el 80% de los retornos), el **ancho medio**
mide cuánta incertidumbre reporta el modelo, y los **cruces** (Q0.9 < Q0.1) delatan mala calibración."""))
cells.append(code(
"""cuant = pred_long[pred_long['Familia'] == 'cuantil'].copy()
cuant['dentro'] = (cuant['ret'] >= cuant['q10']) & (cuant['ret'] <= cuant['q90'])
cuant['ancho'] = cuant['q90'] - cuant['q10']
cobertura = cuant.groupby(['Accion', 'Metodo']).agg(
    cobertura_80=('dentro', 'mean'),
    ancho_medio=('ancho', 'mean'),
    pct_cruces=('ancho', lambda w: float((w < 0).mean())),
).round(3)
cobertura.to_csv('cobertura.csv', encoding='utf-8')
print('Guardado: cobertura.csv. Cobertura ideal ~0.80. pct_cruces > 0 => quantile crossing (mala calibracion).')
cobertura"""))

cells.append(md(
"""## 14. SHAP — interpretación (Objetivo 1)

SHAP es **solo interpretativo** (la selección la hicieron VIF+ADF). Se entrena el modelo sobre la serie
completa y se explica para obtener la relevancia global de cada variable. La Figura 9 de la tesina es el
beeswarm de XGBoost; aquí se genera por acción y se añade la relevancia por año (¿cambia según el
periodo?)."""))
cells.append(code(
"""if DISPONIBLES['shap'] and DISPONIBLES['xgboost']:
    for ticker in TICKERS:
        df = cargar(ticker); X, y = df[FEATURES], df['Target_Retorno_1d']
        m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=0, tree_method='hist').fit(X.values, y.values)
        sv = shap.TreeExplainer(m).shap_values(X.values)
        shap.summary_plot(sv, X, feature_names=FEATURES, show=False)
        plt.title(f'SHAP beeswarm — XGBoost {ticker}'); plt.tight_layout(); plt.show()
else:
    print('SHAP o XGBoost no disponibles; se omite.')"""))

cells.append(code(
"""# ¿Cambia la relevancia segun el periodo? mean|SHAP| relativo por anio (XGBoost, NVDA)
if DISPONIBLES['shap'] and DISPONIBLES['xgboost']:
    df = cargar('NVDA'); X, y = df[FEATURES], df['Target_Retorno_1d']
    m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     random_state=0, tree_method='hist').fit(X.values, y.values)
    sv = shap.TreeExplainer(m).shap_values(X.values)
    sv_abs = pd.DataFrame(np.abs(sv), columns=FEATURES, index=X.index)
    por_anio = sv_abs.groupby(sv_abs.index.year).mean()
    rel = por_anio.div(por_anio.sum(axis=1), axis=0)   # relevancia relativa (suma 1 por anio)
    display(rel.round(3))
else:
    print('SHAP o XGBoost no disponibles; se omite.')"""))

cells.append(md(
"""## Pendientes tras correr

1. Sanity check de resultados (¿el ML supera al Buy & Hold en Sharpe OOS? ¿en qué acción?).
2. ¿Sustituir XGBoost cuantil por la pinball-arctan de Sluijterman (2025) para fidelidad con el Cap. 5?
3. Regenerar la Figura 9 con el set de 9 y reescribir Tabla 2 / §5.4.
4. LSTM con `DeepExplainer` es opcional (caro); por ahora SHAP cubre XGBoost/RF/lineales.
5. Mejora: extraer el motor a `backtest_engine.py` e importarlo en ambos notebooks."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

import os
salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos_ml.ipynb')
with open(salida, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('OK ->', salida, '(', len(cells), 'celdas )')
