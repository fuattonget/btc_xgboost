"""
BTC Trend Analizi & XGBoost Fiyat Tahmin Uygulaması
----------------------------------------------------
- Kraken Public REST API'den canlı BTC/USD 1 dakikalık mum verisi çeker (API key gerekmez,
  ABD dahil çoğu bölgeden erişilebilir; Binance.com ABD IP'lerini 451 hatasıyla engeller)
- En çok kullanılan trade indikatörlerini hesaplar (EMA, SMA, RSI, MACD, Bollinger Bantları,
  Stokastik Osilatör, ATR, OBV, VWAP)
- XGBoost regresyon modeliyle 2 dakika sonrasının kapanış fiyatını tahmin eder
- Her 2 dakikada bir otomatik olarak verileri yeniler ve modeli yeniden eğitir

Çalıştırmak için:
    pip install -r requirements.txt
    streamlit run app.py
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_autorefresh import st_autorefresh
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------
# Sayfa ayarları
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="BTC Trend Analizi & XGBoost Tahmin",
    page_icon="₿",
    layout="wide",
)

SYMBOL = "BTC/USD"
PAIR = "XBTUSD"            # Kraken'de BTC/USD çifti bu şekilde adlandırılır
INTERVAL = "1m"             # Sadece görüntüleme etiketi
KRAKEN_INTERVAL_MIN = 1      # Kraken OHLC aralığı (dakika): 1,5,15,30,60,240,1440,10080,21600
LOOKBACK = 720                # Kraken tek istekte en fazla ~720 mum döndürür
HORIZON = 2               # Kaç dakika sonrası tahmin edilecek (mum sayısı, 1m bar -> 2 dk)
REFRESH_MS = 2 * 60 * 1000  # 2 dakikada bir otomatik yenile

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"

# ----------------------------------------------------------------------------
# Veri çekme
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_klines(pair: str = PAIR, interval_min: int = KRAKEN_INTERVAL_MIN, limit: int = LOOKBACK) -> pd.DataFrame:
    """Kraken'den OHLCV mum verisi çeker (API key gerekmez, ABD dahil çoğu ülkeden erişilebilir)."""
    params = {"pair": pair, "interval": interval_min}
    resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error"):
        raise RuntimeError(f"Kraken API hatası: {payload['error']}")

    result = payload["result"]
    # 'result' içinde hem gerçek çift anahtarı (örn. 'XXBTZUSD') hem de 'last' anahtarı bulunur
    pair_key = next(k for k in result.keys() if k != "last")
    raw = result[pair_key]

    cols = ["open_time", "open", "high", "low", "close", "vwap", "volume", "trades"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    df["close_time"] = df["open_time"] + pd.Timedelta(minutes=interval_min)
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    df.set_index("open_time", inplace=True)
    return df.tail(limit)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_ticker_24h(pair: str = PAIR) -> dict:
    """Kraken Ticker endpoint'inden 24 saatlik özet veriyi çeker ve Binance ile uyumlu
    basit bir sözlük formatına (priceChangePercent, volume) dönüştürür."""
    resp = requests.get(KRAKEN_TICKER_URL, params={"pair": pair}, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error"):
        raise RuntimeError(f"Kraken API hatası: {payload['error']}")

    result = payload["result"]
    pair_key = next(iter(result.keys()))
    t = result[pair_key]

    last_price = float(t["c"][0])       # son işlem fiyatı
    today_open = float(t["o"])           # bugünkü açılış fiyatı
    volume_24h = float(t["v"][1])        # 24 saatlik hacim

    change_pct = (last_price - today_open) / today_open * 100 if today_open else 0.0

    return {"priceChangePercent": change_pct, "volume": volume_24h}


# ----------------------------------------------------------------------------
# Teknik indikatörler (harici kütüphaneye ihtiyaç duymadan hesaplanır)
# ----------------------------------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def stochastic_oscillator(high, low, close, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k.fillna(50), d.fillna(50)


def atr(high, low, close, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def vwap(high, low, close, volume) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisinden indikatörleri ve model özelliklerini üretir."""
    out = df.copy()

    out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["sma_20"] = out["close"].rolling(20).mean()

    out["rsi_14"] = rsi(out["close"], 14)

    macd_line, macd_signal, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    bb_upper, bb_mid, bb_lower = bollinger_bands(out["close"])
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid

    stoch_k, stoch_d = stochastic_oscillator(out["high"], out["low"], out["close"])
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    out["atr_14"] = atr(out["high"], out["low"], out["close"])
    out["obv"] = obv(out["close"], out["volume"])
    out["vwap"] = vwap(out["high"], out["low"], out["close"], out["volume"])

    # Getiri / momentum tabanlı ek özellikler
    out["return_1"] = out["close"].pct_change(1)
    out["return_5"] = out["close"].pct_change(5)
    out["return_15"] = out["close"].pct_change(15)
    out["volatility_15"] = out["return_1"].rolling(15).std()

    # Gecikmeli (lag) kapanış fiyatları - zaman serisi bağlamı için
    for lag in [1, 2, 3, 5, 10]:
        out[f"close_lag_{lag}"] = out["close"].shift(lag)

    return out


FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "sma_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "stoch_k", "stoch_d", "atr_14", "obv", "vwap",
    "return_1", "return_5", "return_15", "volatility_15",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
    "volume",
]


# ----------------------------------------------------------------------------
# Model eğitimi
# ----------------------------------------------------------------------------
def train_model(feat_df: pd.DataFrame, horizon: int = HORIZON):
    """XGBoost regresyon modelini eğitir; hedef = `horizon` mum sonraki kapanış fiyatı."""
    data = feat_df.copy()
    data["target"] = data["close"].shift(-horizon)
    data = data.dropna(subset=FEATURE_COLS + ["target"])

    if len(data) < 50:
        return None, None, None

    X = data[FEATURE_COLS]
    y = data["target"]

    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = {}
    if len(X_test) > 0:
        preds_test = model.predict(X_test)
        metrics["mae"] = mean_absolute_error(y_test, preds_test)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, preds_test))
        metrics["n_test"] = len(X_test)

    # Son satırdan geleceğe dair tahmin (target NaN olan en güncel satır)
    last_row = feat_df.dropna(subset=FEATURE_COLS).iloc[[-1]][FEATURE_COLS]
    future_pred = model.predict(last_row)[0]

    return model, metrics, future_pred


def trend_signal(latest: pd.Series) -> tuple[str, str]:
    """Basit kural tabanlı trend/sinyal özeti (indikatör kombinasyonu)."""
    score = 0
    reasons = []

    if latest["ema_9"] > latest["ema_21"] > latest["ema_50"]:
        score += 2
        reasons.append("EMA9 > EMA21 > EMA50 (güçlü yükseliş dizilimi)")
    elif latest["ema_9"] < latest["ema_21"] < latest["ema_50"]:
        score -= 2
        reasons.append("EMA9 < EMA21 < EMA50 (güçlü düşüş dizilimi)")

    if latest["macd"] > latest["macd_signal"]:
        score += 1
        reasons.append("MACD sinyal çizgisinin üzerinde")
    else:
        score -= 1
        reasons.append("MACD sinyal çizgisinin altında")

    if latest["rsi_14"] > 70:
        score -= 1
        reasons.append(f"RSI {latest['rsi_14']:.1f} (aşırı alım)")
    elif latest["rsi_14"] < 30:
        score += 1
        reasons.append(f"RSI {latest['rsi_14']:.1f} (aşırı satım)")

    if latest["close"] > latest["bb_upper"]:
        score -= 1
        reasons.append("Fiyat üst Bollinger bandının üzerinde")
    elif latest["close"] < latest["bb_lower"]:
        score += 1
        reasons.append("Fiyat alt Bollinger bandının altında")

    if latest["stoch_k"] > 80:
        score -= 1
        reasons.append("Stokastik aşırı alım bölgesinde")
    elif latest["stoch_k"] < 20:
        score += 1
        reasons.append("Stokastik aşırı satım bölgesinde")

    if score >= 2:
        label = "GÜÇLÜ YÜKSELİŞ"
    elif score == 1:
        label = "HAFİF YÜKSELİŞ"
    elif score == 0:
        label = "NÖTR"
    elif score == -1:
        label = "HAFİF DÜŞÜŞ"
    else:
        label = "GÜÇLÜ DÜŞÜŞ"

    return label, " • ".join(reasons)


# ----------------------------------------------------------------------------
# Tahmin doğruluk takibi (canlı backtest)
# ----------------------------------------------------------------------------
def log_new_prediction(feat_df: pd.DataFrame, latest: pd.Series, current_price: float,
                        future_pred: float, horizon: int = HORIZON) -> None:
    """Yeni yapılan tahmini session_state'e kaydeder (aynı bar için tekrar kaydetmez)."""
    log = st.session_state.setdefault("prediction_log", [])

    pred_time = latest.name  # bu barın zaman damgası (tz-aware)
    if log and log[-1]["pred_time"] == pred_time:
        return  # bu bar için zaten kayıt var, tekrar ekleme

    if future_pred is None:
        return

    target_time = pred_time + pd.Timedelta(minutes=horizon)
    log.append({
        "pred_time": pred_time,
        "target_time": target_time,
        "price_at_pred": float(current_price),
        "predicted_price": float(future_pred),
        "actual_price": None,
        "actual_time": None,
        "resolved": False,
    })
    # Log'un çok büyümesini önle
    if len(log) > 300:
        del log[: len(log) - 300]


def resolve_pending_predictions(feat_df: pd.DataFrame) -> None:
    """Hedef zamanı geçmiş, henüz sonuçlanmamış tahminleri gerçekleşen fiyatla eşleştirir."""
    log = st.session_state.get("prediction_log", [])
    for entry in log:
        if entry["resolved"]:
            continue
        matching = feat_df[feat_df.index >= entry["target_time"]]
        if matching.empty:
            continue
        actual_row = matching.iloc[0]
        entry["actual_price"] = float(actual_row["close"])
        entry["actual_time"] = matching.index[0]
        entry["resolved"] = True


def prediction_accuracy_summary() -> pd.DataFrame | None:
    """Sonuçlanmış tahminlerden bir özet tablo üretir."""
    log = st.session_state.get("prediction_log", [])
    resolved = [e for e in log if e["resolved"]]
    if not resolved:
        return None

    rows = []
    for e in resolved[-30:][::-1]:  # en yeni en üstte, son 30 tahmin
        error_abs = abs(e["actual_price"] - e["predicted_price"])
        error_pct = error_abs / e["actual_price"] * 100
        pred_dir = np.sign(e["predicted_price"] - e["price_at_pred"])
        actual_dir = np.sign(e["actual_price"] - e["price_at_pred"])
        direction_ok = "✅" if (pred_dir == actual_dir or actual_dir == 0) else "❌"
        rows.append({
            "Tahmin Zamanı (UTC)": e["pred_time"].strftime("%H:%M:%S"),
            "O Anki Fiyat": f"${e['price_at_pred']:,.2f}",
            "Tahmin Edilen": f"${e['predicted_price']:,.2f}",
            "Gerçekleşen": f"${e['actual_price']:,.2f}",
            "Hata": f"${error_abs:,.2f} ({error_pct:.3f}%)",
            "Yön Doğru mu?": direction_ok,
        })
    return pd.DataFrame(rows)


def render_countdown_and_forecast_widget(target_time: pd.Timestamp, current_price: float,
                                          future_pred: float, refresh_ms: int) -> None:
    """Sonraki yenilemeye kalan süreyi sayan ve modelin ne zamana kadar
    yükseliş/düşüş beklediğini gösteren küçük bir HTML/JS widget'ı render eder."""
    target_iso = target_time.isoformat()
    refresh_seconds = refresh_ms // 1000

    if future_pred > current_price:
        direction_word, arrow, color = "YÜKSELİŞ", "▲", "#16a34a"
    elif future_pred < current_price:
        direction_word, arrow, color = "DÜŞÜŞ", "▼", "#dc2626"
    else:
        direction_word, arrow, color = "YATAY", "→", "#6b7280"

    pct_change = (future_pred - current_price) / current_price * 100

    html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; font-family:inherit;">
      <div style="flex:1; min-width:220px; padding:14px 18px; border-radius:12px;
                  border:1px solid rgba(128,128,128,0.3); background:rgba(128,128,128,0.06);">
        <div style="font-size:13px; opacity:0.7; margin-bottom:4px;">⏱ Sonraki Yenilemeye Kalan Süre</div>
        <div id="countdown" style="font-size:26px; font-weight:700;">--:--</div>
      </div>
      <div style="flex:2; min-width:280px; padding:14px 18px; border-radius:12px;
                  border:1px solid rgba(128,128,128,0.3); background:rgba(128,128,128,0.06);">
        <div style="font-size:13px; opacity:0.7; margin-bottom:4px;">📊 Model Ne Bekliyor?</div>
        <div style="font-size:18px; font-weight:700; color:{color};">
          {arrow} <span id="target-time">--:--:--</span> saatine (yerel saat) kadar {direction_word}
          <span style="font-weight:500; font-size:15px;">({pct_change:+.3f}%)</span>
        </div>
      </div>
    </div>
    <script>
      let remaining = {refresh_seconds};
      const countdownEl = document.getElementById('countdown');
      function tick() {{
        const m = Math.floor(remaining / 60).toString().padStart(2, '0');
        const s = (remaining % 60).toString().padStart(2, '0');
        countdownEl.innerText = m + ':' + s;
        if (remaining > 0) {{
          remaining -= 1;
          setTimeout(tick, 1000);
        }} else {{
          countdownEl.innerText = 'Yenileniyor...';
        }}
      }}
      tick();

      const targetDate = new Date("{target_iso}");
      document.getElementById('target-time').innerText =
        targetDate.toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit', second: '2-digit'}});
    </script>
    """
    components.html(html, height=110)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st_autorefresh(interval=REFRESH_MS, key="auto_refresh")

st.title("₿ BTC/USD Trend Analizi & XGBoost Fiyat Tahmini")
st.caption(
    f"Veri kaynağı: Kraken Public API • Mum aralığı: {INTERVAL} • "
    f"Tahmin ufku: {HORIZON} dakika • Otomatik yenileme: 2 dakikada bir"
)

with st.spinner("Canlı BTC verisi çekiliyor..."):
    try:
        raw_df = fetch_klines()
        ticker = fetch_ticker_24h()
        fetch_error = None
    except Exception as exc:  # noqa: BLE001
        raw_df, ticker, fetch_error = None, None, str(exc)

if fetch_error:
    st.error(
        "Kraken API'ye ulaşılamadı. İnternet bağlantınızı kontrol edin "
        f"veya bir süre sonra tekrar deneyin.\n\nHata detayı: {fetch_error}"
    )
    st.stop()

feat_df = build_features(raw_df)
latest = feat_df.iloc[-1]

with st.spinner("XGBoost modeli eğitiliyor ve tahmin hesaplanıyor..."):
    model, metrics, future_pred = train_model(feat_df)

# ---- Tahmin doğruluk takibi: önce geçmiş tahminleri sonuçlandır, sonra yenisini kaydet
resolve_pending_predictions(feat_df)
log_new_prediction(feat_df, latest, latest["close"], future_pred)

# ---- Üst metrik satırı --------------------------------------------------
current_price = latest["close"]
change_24h_pct = float(ticker["priceChangePercent"])
volume_24h = float(ticker["volume"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Güncel Fiyat", f"${current_price:,.2f}", f"{change_24h_pct:+.2f}% (24s)")

if future_pred is not None:
    delta = future_pred - current_price
    delta_pct = delta / current_price * 100
    col2.metric(
        f"{HORIZON} Dakika Sonrası Tahmin",
        f"${future_pred:,.2f}",
        f"{delta_pct:+.3f}%",
    )
else:
    col2.metric(f"{HORIZON} Dakika Sonrası Tahmin", "Yetersiz veri", "—")

label, reasons = trend_signal(latest)
col3.metric("Kural Tabanlı Trend Sinyali", label)

if metrics:
    col4.metric("Model Hata Payı (MAE)", f"${metrics['mae']:,.2f}", f"RMSE: ${metrics['rmse']:,.2f}")
else:
    col4.metric("Model Hata Payı (MAE)", "—")

st.info(f"**Sinyal gerekçeleri:** {reasons}")

if future_pred is not None:
    target_time = latest.name + pd.Timedelta(minutes=HORIZON)
    render_countdown_and_forecast_widget(target_time, current_price, future_pred, REFRESH_MS)

st.caption(
    f"Son güncelleme (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} — "
    "Bu uygulama yatırım tavsiyesi değildir, eğitim/analiz amaçlıdır."
)

st.divider()

# ---- Fiyat + indikatör grafikleri ---------------------------------------
plot_df = feat_df.tail(200)

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.45, 0.2, 0.2, 0.15],
    subplot_titles=("Fiyat, EMA & Bollinger Bantları", "RSI (14)", "MACD", "Hacim & OBV"),
)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["open"], high=plot_df["high"],
    low=plot_df["low"], close=plot_df["close"], name="BTC/USD",
), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema_9"], name="EMA 9", line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema_21"], name="EMA 21", line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema_50"], name="EMA 50", line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"], name="BB Üst", line=dict(width=1, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"], name="BB Alt", line=dict(width=1, dash="dot")), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi_14"], name="RSI 14", line=dict(color="orange")), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd"], name="MACD", line=dict(color="blue")), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_signal"], name="Sinyal", line=dict(color="red")), row=3, col=1)
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["macd_hist"], name="MACD Hist"), row=3, col=1)

fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["volume"], name="Hacim"), row=4, col=1)

fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---- Tahmin doğruluk takibi (canlı backtest) -----------------------------
st.subheader("🎯 Tahmin Doğruluk Takibi (Canlı Backtest)")
st.caption(
    "Her yenilemede, bir önceki döngüde yapılan tahmin artık gerçekleşen fiyatla "
    "karşılaştırılıp burada gösterilir. Uygulamayı açık bıraktıkça bu liste büyür."
)

resolved_log = [e for e in st.session_state.get("prediction_log", []) if e["resolved"]]

if not resolved_log:
    st.warning(
        "Henüz sonuçlanmış bir tahmin yok. İlk tahminin sonucu, tahmin edilen "
        f"{HORIZON} dakikalık süre dolduktan sonraki ilk yenilemede burada görünecek."
    )
else:
    errors_pct = [
        abs(e["actual_price"] - e["predicted_price"]) / e["actual_price"] * 100
        for e in resolved_log
    ]
    errors_abs = [abs(e["actual_price"] - e["predicted_price"]) for e in resolved_log]
    direction_hits = [
        1 if (np.sign(e["predicted_price"] - e["price_at_pred"])
              == np.sign(e["actual_price"] - e["price_at_pred"])
              or e["actual_price"] == e["price_at_pred"]) else 0
        for e in resolved_log
    ]

    acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
    acc_col1.metric("Sonuçlanan Tahmin Sayısı", len(resolved_log))
    acc_col2.metric("Ortalama Mutlak Hata", f"${np.mean(errors_abs):,.2f}")
    acc_col3.metric("Ortalama Yüzde Hata", f"{np.mean(errors_pct):.3f}%")
    acc_col4.metric("Yön İsabet Oranı", f"{np.mean(direction_hits) * 100:.1f}%")

    acc_table = prediction_accuracy_summary()
    st.dataframe(acc_table, hide_index=True, use_container_width=True)

st.divider()

# ---- Ek indikatörler tablosu ---------------------------------------------
st.subheader("Anlık İndikatör Değerleri")
indicator_table = pd.DataFrame({
    "İndikatör": [
        "EMA 9 / 21 / 50", "SMA 20", "RSI (14)", "MACD / Sinyal",
        "Bollinger Üst / Orta / Alt", "Stokastik %K / %D", "ATR (14)", "VWAP", "OBV",
    ],
    "Değer": [
        f"{latest['ema_9']:.2f} / {latest['ema_21']:.2f} / {latest['ema_50']:.2f}",
        f"{latest['sma_20']:.2f}",
        f"{latest['rsi_14']:.2f}",
        f"{latest['macd']:.2f} / {latest['macd_signal']:.2f}",
        f"{latest['bb_upper']:.2f} / {latest['bb_mid']:.2f} / {latest['bb_lower']:.2f}",
        f"{latest['stoch_k']:.2f} / {latest['stoch_d']:.2f}",
        f"{latest['atr_14']:.2f}",
        f"{latest['vwap']:.2f}",
        f"{latest['obv']:.0f}",
    ],
})
st.dataframe(indicator_table, hide_index=True, use_container_width=True)

# ---- Özellik önem sırası ---------------------------------------------------
if model is not None:
    st.subheader("XGBoost Özellik Önem Sırası (Model Neye Bakıyor?)")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fig_imp = go.Figure(go.Bar(x=importance.values[:12][::-1], y=importance.index[:12][::-1], orientation="h"))
    fig_imp.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

st.caption(
    "Not: Model her 2 dakikalık yenilemede en güncel 1000 mumluk pencere üzerinde yeniden eğitilir. "
    "Kripto piyasaları yüksek oynaklığa sahiptir; bu araç finansal tavsiye değildir."
)