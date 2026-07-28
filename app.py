"""
BTC Trend Analizi & XGBoost Fiyat Tahmin Uygulaması
----------------------------------------------------
- Kraken Public REST API'den canlı BTC/USD 1 dakikalık mum verisi çeker (API key gerekmez,
  ABD dahil çoğu bölgeden erişilebilir; Binance.com ABD IP'lerini 451 hatasıyla engeller)
- En çok kullanılan trade indikatörlerini hesaplar (EMA, SMA, RSI, MACD, Bollinger Bantları,
  Stokastik Osilatör, ATR, OBV, VWAP)
- 15 dakikalık üst zaman dilimi verisinden "rejim" (trend bağlamı) özellikleri türetir
  (Kraken'in 1 dakikalık geçmişi ~12 saatle sınırlı; 15dk mumlarla ~7.5 günlük bağlam eklenir)
- XGBoost regresyon modeliyle 2 dakika sonrasının GETİRİSİNİ (log-return) tahmin eder ve
  fiyata çevirir; ayrıca yön (yukarı/aşağı) için ayrı bir XGBoost sınıflandırma modeli eğitir
- Model performansını naive baseline (tahmin = şu anki fiyat) ile karşılaştırır
- Walk-forward (embargo'lu) train/test ayrımı ve TimeSeriesSplit ile periyodik
  hiperparametre seçimi yapar
- Tahmin geçmişini SQLite'a kalıcı olarak kaydeder (uygulama/sunucu yeniden başlasa bile kaybolmaz)
- Her gerçek model eğitiminde (cache miss olduğunda) kullanılan hiperparametreleri, test
  seti hata oranlarını (MAE/RMSE/naive'e karşı edge, yön isabeti) ve modelin en çok neye
  baktığını (özellik önem sırası) SQLite'a loglar
- Sonuçlanmış tahminleri ve eğitim loglarını periyodik olarak GitHub'daki ayrı bir `data`
  branch'ine CSV olarak senkronize eder (deploy edilen `main` branch'ini TETİKLEMEZ) —
  böylece canlı sonuçlar repo üzerinden dışarıdan da takip edilebilir. `GITHUB_TOKEN`
  secret'ı tanımlı değilse bu adım sessizce atlanır, uygulamanın geri kalanını etkilemez.
- Her 2 dakikada bir otomatik olarak verileri yeniler ve modeli yeniden eğitir

Çalıştırmak için:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from streamlit_autorefresh import st_autorefresh
from xgboost import XGBClassifier, XGBRegressor

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
LOOKBACK = 720                # Kraken tek istekte en fazla ~720 mum döndürür (~12 saat @1m)
HORIZON = 2               # Kaç dakika sonrası tahmin edilecek (mum sayısı, 1m bar -> 2 dk)
REFRESH_MS = 2 * 60 * 1000  # 2 dakikada bir otomatik yenile

REGIME_INTERVAL_MIN = 15     # Üst zaman dilimi (rejim/bağlam özellikleri için)
REGIME_LOOKBACK = 720          # ~7.5 gün @15m — Kraken 1m'de veremediği uzun geçmişi kısmen telafi eder

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"

DB_PATH = Path(__file__).resolve().parent / "predictions.db"

GITHUB_REPO = "fuattonget/btc_xgboost"
GITHUB_DATA_BRANCH = "data"
GITHUB_DATA_PATH = "data/predictions_log.csv"
GITHUB_TRAINING_PATH = "data/training_log.csv"
GITHUB_SYNC_MIN_INTERVAL_MIN = 10

DEFAULT_PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 400},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 150},
]

# ----------------------------------------------------------------------------
# Veri çekme
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_klines(pair: str = PAIR, interval_min: int = KRAKEN_INTERVAL_MIN, limit: int = LOOKBACK) -> pd.DataFrame:
    """Kraken'den OHLCV mum verisi çeker (API key gerekmez, ABD dahil çoğu ülkeden erişilebilir).

    Not: Kraken'in `since` parametresi test edildi — geçmişe ne kadar gidilirse gidilsin
    her interval için sabit ~720 mumluk pencere döner (1m'de ~12 saat). Yani sayfalama ile
    daha fazla 1 dakikalık geçmiş elde etmek mümkün değil; bu proje bu sınırı `REGIME_INTERVAL_MIN`
    ile telafi ediyor (bkz. build_regime_features).
    """
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


def build_regime_features(regime_df: pd.DataFrame) -> pd.DataFrame:
    """15 dakikalık üst zaman diliminden, 1 dakikalık modele "büyük resmi" (rejim/bağlam)
    kazandıran birkaç özellik türetir. Kraken 1 dakikalık geçmişi ~12 saatle sınırlı olduğu
    için modelin son birkaç saatin ötesini görmesinin tek yolu bu üst zaman dilimi."""
    out = regime_df.copy()
    ema_20 = out["close"].ewm(span=20, adjust=False).mean()
    ema_50 = out["close"].ewm(span=50, adjust=False).mean()
    out["regime_close_vs_ema50"] = (out["close"] - ema_50) / ema_50
    out["regime_rsi_14"] = rsi(out["close"], 14)
    out["regime_trend"] = np.where(ema_20 > ema_50, 1.0, -1.0)
    return out[["regime_close_vs_ema50", "regime_rsi_14", "regime_trend"]].sort_index()


def build_features(df: pd.DataFrame, regime_feat: pd.DataFrame | None = None) -> pd.DataFrame:
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

    if regime_feat is not None and not regime_feat.empty:
        out = pd.merge_asof(
            out.sort_index(), regime_feat,
            left_index=True, right_index=True, direction="backward",
        )
    else:
        for c in ["regime_close_vs_ema50", "regime_rsi_14", "regime_trend"]:
            out[c] = np.nan

    return out


FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "sma_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "stoch_k", "stoch_d", "atr_14", "obv", "vwap",
    "return_1", "return_5", "return_15", "volatility_15",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
    "volume",
    "regime_close_vs_ema50", "regime_rsi_14", "regime_trend",
]


# ----------------------------------------------------------------------------
# Model eğitimi
# ----------------------------------------------------------------------------
def prepare_training_data(feat_df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Hedef değişkenleri üretir. Ham fiyat yerine LOG-RETURN hedeflenir: modelin
    "şimdiki fiyatı kopyalayarak" düşük hata elde etmesini engeller, çünkü lag/EMA gibi
    fiyata çok yakın özellikler artık hedefi (getiri) doğrudan açıklamıyor."""
    data = feat_df.copy()
    data["target_price"] = data["close"].shift(-horizon)
    data["target_return"] = np.log(data["target_price"] / data["close"])
    data["target_direction"] = (data["target_return"] > 0).astype(int)
    data = data.dropna(subset=FEATURE_COLS + ["target_price", "target_return"])
    return data


def time_series_split_with_embargo(data: pd.DataFrame, horizon: int, test_frac: float = 0.15):
    """Kronolojik train/test ayrımı + embargo (purge) boşluğu.

    Hedef `horizon` bar ileriye baktığı için, split noktasına en yakın train satırlarının
    hedefleri test bölgesindeki fiyatlarla örtüşür. Aradaki `horizon` satırı train setinden
    çıkararak bu örtüşmeyi (optimistik sızıntıyı) engelliyoruz.
    """
    n = len(data)
    split = int(n * (1 - test_frac))
    train_end = max(split - horizon, int(n * 0.5))
    train = data.iloc[:train_end]
    test = data.iloc[split:]
    return train, test


@st.cache_data(ttl=1800, show_spinner=False)
def select_hyperparams(feat_df: pd.DataFrame, horizon: int = HORIZON, n_splits: int = 4):
    """TimeSeriesSplit (gap=horizon ile embargo'lu) walk-forward CV ile küçük bir grid
    üzerinde en iyi hiperparametreleri seçer. Sonuç 30 dakika cache'lenir; her 2 dakikalık
    otomatik yenilemede tekrar grid-search çalıştırmak gereksiz ve maliyetlidir."""
    data = prepare_training_data(feat_df, horizon)
    X = data[FEATURE_COLS]
    y = data["target_return"]

    n_splits_eff = min(n_splits, max(2, len(data) // 100))
    tscv = TimeSeriesSplit(n_splits=n_splits_eff, gap=horizon)

    rows = []
    for params in DEFAULT_PARAM_GRID:
        fold_maes = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            m = XGBRegressor(
                objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=42, n_jobs=-1, **params,
            )
            m.fit(X_tr, y_tr)
            pred_return = m.predict(X_te)
            current_price = data["close"].iloc[test_idx].values
            pred_price = current_price * np.exp(pred_return)
            actual_price = data["target_price"].iloc[test_idx].values
            fold_maes.append(mean_absolute_error(actual_price, pred_price))
        rows.append({**params, "cv_mae": float(np.mean(fold_maes))})

    report = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    best_row = report.iloc[0]
    best_params = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "n_estimators": int(best_row["n_estimators"]),
    }
    return best_params, report


@st.cache_data(ttl=90, show_spinner=False)
def train_model(feat_df: pd.DataFrame, horizon: int, params: dict):
    """Regresyon (fiyat/getiri) + sınıflandırma (yön) modellerini eğitir; naive baseline'a
    karşı test seti performansını ölçer. `feat_df`/`horizon`/`params` değişmediği sürece
    (yani yeni mum verisi gelmediği sürece) cache'den döner — her Streamlit rerun'unda
    sıfırdan eğitim yapılmaz."""
    data = prepare_training_data(feat_df, horizon)
    if len(data) < 100:
        return None

    train, test = time_series_split_with_embargo(data, horizon)
    if len(train) < 50 or len(test) < 5:
        return None

    X_train, y_train = train[FEATURE_COLS], train["target_return"]
    X_test = test[FEATURE_COLS]

    reg = XGBRegressor(
        objective="reg:squarederror", subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1, **params,
    )
    reg.fit(X_train, y_train)

    pred_return_test = reg.predict(X_test)
    current_price_test = test["close"].values
    pred_price_test = current_price_test * np.exp(pred_return_test)
    actual_price_test = test["target_price"].values

    mae_model = float(mean_absolute_error(actual_price_test, pred_price_test))
    rmse_model = float(np.sqrt(mean_squared_error(actual_price_test, pred_price_test)))
    mae_naive = float(mean_absolute_error(actual_price_test, current_price_test))
    edge_vs_naive_pct = (mae_naive - mae_model) / mae_naive * 100 if mae_naive else 0.0

    pred_dir = np.sign(pred_price_test - current_price_test)
    actual_dir = np.sign(actual_price_test - current_price_test)
    direction_acc = float(np.mean((pred_dir == actual_dir) | (actual_dir == 0)))

    clf = XGBClassifier(
        max_depth=params["max_depth"], learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"], subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    clf.fit(train[FEATURE_COLS], train["target_direction"])
    clf_test_acc = float(clf.score(X_test, test["target_direction"]))

    last_row = feat_df.dropna(subset=FEATURE_COLS).iloc[[-1]][FEATURE_COLS]
    current_price = float(feat_df["close"].iloc[-1])
    future_return = float(reg.predict(last_row)[0])
    future_pred = current_price * np.exp(future_return)
    future_up_prob = float(clf.predict_proba(last_row)[0][1])

    metrics = {
        "mae": mae_model,
        "rmse": rmse_model,
        "mae_naive": mae_naive,
        "edge_vs_naive_pct": edge_vs_naive_pct,
        "n_test": int(len(test)),
        "direction_acc": direction_acc,
        "clf_test_acc": clf_test_acc,
    }

    reg_importance = pd.Series(reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(15)
    clf_importance = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(15)
    log_training_run(horizon, params, len(train), metrics, reg_importance, clf_importance)

    return {
        "model": reg,
        "clf_model": clf,
        "metrics": metrics,
        "future_pred": future_pred,
        "future_up_prob": future_up_prob,
    }


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
# Tahmin doğruluk takibi (canlı backtest) — SQLite'a kalıcı olarak yazılır
# ----------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            pred_time TEXT PRIMARY KEY,
            target_time TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            price_at_pred REAL NOT NULL,
            predicted_price REAL NOT NULL,
            actual_price REAL,
            actual_time TEXT,
            resolved INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            run_time TEXT PRIMARY KEY,
            horizon INTEGER NOT NULL,
            max_depth INTEGER NOT NULL,
            learning_rate REAL NOT NULL,
            n_estimators INTEGER NOT NULL,
            n_train INTEGER NOT NULL,
            n_test INTEGER NOT NULL,
            mae REAL NOT NULL,
            rmse REAL NOT NULL,
            mae_naive REAL NOT NULL,
            edge_vs_naive_pct REAL NOT NULL,
            direction_acc REAL NOT NULL,
            clf_test_acc REAL NOT NULL,
            reg_feature_importance TEXT NOT NULL,
            clf_feature_importance TEXT NOT NULL
        )
        """
    )
    return conn


def log_training_run(horizon: int, params: dict, n_train: int, metrics: dict,
                      reg_importance: pd.Series, clf_importance: pd.Series) -> None:
    """Her GERÇEK model eğitiminde (train_model'in cache miss olduğu her seferde) çağrılır:
    kullanılan hiperparametreleri, test seti hata oranlarını ve her iki modelin en çok
    hangi özelliklere baktığını (importance) SQLite'a kalıcı olarak kaydeder."""
    run_time = pd.Timestamp.now(tz="UTC").isoformat()
    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO training_runs "
            "(run_time, horizon, max_depth, learning_rate, n_estimators, n_train, n_test, "
            " mae, rmse, mae_naive, edge_vs_naive_pct, direction_acc, clf_test_acc, "
            " reg_feature_importance, clf_feature_importance) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_time, horizon,
                params["max_depth"], params["learning_rate"], params["n_estimators"],
                n_train, metrics["n_test"],
                metrics["mae"], metrics["rmse"], metrics["mae_naive"], metrics["edge_vs_naive_pct"],
                metrics["direction_acc"], metrics["clf_test_acc"],
                json.dumps(reg_importance.round(6).to_dict()),
                json.dumps(clf_importance.round(6).to_dict()),
            ),
        )
    conn.close()


def load_training_runs(limit: int | None = None) -> pd.DataFrame:
    conn = get_db_connection()
    query = "SELECT * FROM training_runs ORDER BY run_time DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def log_new_prediction(latest_time: pd.Timestamp, current_price: float,
                        future_pred: float, horizon: int = HORIZON) -> None:
    """Yeni yapılan tahmini SQLite'a kaydeder (aynı bar için tekrar kaydetmez, uygulama
    yeniden başlasa bile geçmiş korunur)."""
    if future_pred is None:
        return
    target_time = latest_time + pd.Timedelta(minutes=horizon)
    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO predictions "
            "(pred_time, target_time, horizon, price_at_pred, predicted_price) "
            "VALUES (?, ?, ?, ?, ?)",
            (latest_time.isoformat(), target_time.isoformat(), horizon,
             float(current_price), float(future_pred)),
        )
    conn.close()


def resolve_pending_predictions(feat_df: pd.DataFrame) -> None:
    """Hedef zamanı geçmiş, henüz sonuçlanmamış tahminleri gerçekleşen fiyatla eşleştirir."""
    conn = get_db_connection()
    pending = conn.execute(
        "SELECT pred_time, target_time FROM predictions WHERE resolved = 0"
    ).fetchall()
    for pred_time_str, target_time_str in pending:
        target_time = pd.Timestamp(target_time_str)
        matching = feat_df[feat_df.index >= target_time]
        if matching.empty:
            continue
        actual_row = matching.iloc[0]
        with conn:
            conn.execute(
                "UPDATE predictions SET actual_price = ?, actual_time = ?, resolved = 1 "
                "WHERE pred_time = ?",
                (float(actual_row["close"]), matching.index[0].isoformat(), pred_time_str),
            )
    conn.close()


def load_resolved_predictions(limit: int | None = None) -> pd.DataFrame:
    conn = get_db_connection()
    query = "SELECT * FROM predictions WHERE resolved = 1 ORDER BY pred_time DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def _get_sync_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_sync_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO sync_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def _push_csv_to_github(token: str, path: str, df: pd.DataFrame, message: str) -> bool:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    content_b64 = base64.b64encode(csv_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

    get_resp = requests.get(api_url, headers=headers, params={"ref": GITHUB_DATA_BRANCH}, timeout=10)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

    payload = {"message": message, "content": content_b64, "branch": GITHUB_DATA_BRANCH}
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
    return put_resp.status_code in (200, 201)


def sync_logs_to_github() -> None:
    """Sonuçlanmış tahmin geçmişini VE model eğitim loglarını (hiperparametreler, hata
    oranları, özellik önem sırası) GitHub'daki ayrı `data` branch'ine iki ayrı CSV olarak
    push eder: `data/predictions_log.csv` ve `data/training_log.csv`.

    Bilerek deploy edilen `main` DEĞİL, ayrı bir `data` branch'i hedeflenir: Streamlit Cloud
    yalnızca app'e bağlı branch'e (main) push geldiğinde otomatik redeploy tetikler; `data`
    branch'ine yazmak canlı uygulamayı yeniden başlatmaz / SQLite'ı sıfırlamaz.

    `st.secrets["GITHUB_TOKEN"]` tanımlı değilse (örn. yerel geliştirmede) bu fonksiyon
    sessizce hiçbir şey yapmaz — özellik tamamen opsiyoneldir ve eksikliği uygulamanın
    geri kalanını etkilemez. Ağ/API hataları da yutulur (best-effort telemetri).
    """
    try:
        token = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        return

    conn = get_db_connection()
    last_synced_str = _get_sync_state(conn, "last_synced_at")
    now = pd.Timestamp.now(tz="UTC")
    if last_synced_str is not None:
        last_synced = pd.Timestamp(last_synced_str)
        if (now - last_synced) < pd.Timedelta(minutes=GITHUB_SYNC_MIN_INTERVAL_MIN):
            conn.close()
            return

    resolved = pd.read_sql_query(
        "SELECT * FROM predictions WHERE resolved = 1 ORDER BY pred_time ASC", conn
    )
    training = pd.read_sql_query("SELECT * FROM training_runs ORDER BY run_time ASC", conn)
    conn.close()

    if resolved.empty and training.empty:
        return

    try:
        ok = True
        if not resolved.empty:
            ok = _push_csv_to_github(
                token, GITHUB_DATA_PATH, resolved,
                f"Update prediction log ({len(resolved)} resolved predictions)",
            ) and ok
        if not training.empty:
            ok = _push_csv_to_github(
                token, GITHUB_TRAINING_PATH, training,
                f"Update training log ({len(training)} runs)",
            ) and ok
        if ok:
            sync_conn = get_db_connection()
            _set_sync_state(sync_conn, "last_synced_at", now.isoformat())
            sync_conn.close()
    except Exception:
        pass  # best-effort: senkronizasyon hatası canlı uygulamayı etkilemesin


def render_countdown_and_forecast_widget(target_time: pd.Timestamp, current_price: float,
                                          future_pred: float, refresh_ms: int,
                                          up_prob: float | None = None) -> None:
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
    prob_suffix = f" &middot; sınıflandırıcı: %{up_prob * 100:.0f} yukarı" if up_prob is not None else ""

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
          <span style="font-weight:500; font-size:15px;">({pct_change:+.3f}%{prob_suffix})</span>
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
        raw_df = fetch_klines(PAIR, KRAKEN_INTERVAL_MIN, LOOKBACK)
        regime_df = fetch_klines(PAIR, REGIME_INTERVAL_MIN, REGIME_LOOKBACK)
        ticker = fetch_ticker_24h()
        fetch_error = None
    except Exception as exc:  # noqa: BLE001
        raw_df, regime_df, ticker, fetch_error = None, None, None, str(exc)

if fetch_error:
    st.error(
        "Kraken API'ye ulaşılamadı. İnternet bağlantınızı kontrol edin "
        f"veya bir süre sonra tekrar deneyin.\n\nHata detayı: {fetch_error}"
    )
    st.stop()

regime_feat = build_regime_features(regime_df)
feat_df = build_features(raw_df, regime_feat)
latest = feat_df.iloc[-1]

with st.spinner("Hiperparametreler seçiliyor (walk-forward CV)..."):
    best_params, cv_report = select_hyperparams(feat_df, HORIZON)

with st.spinner("XGBoost modelleri eğitiliyor ve tahmin hesaplanıyor..."):
    result = train_model(feat_df, HORIZON, best_params)

model = result["model"] if result else None
metrics = result["metrics"] if result else None
future_pred = result["future_pred"] if result else None
future_up_prob = result["future_up_prob"] if result else None

# ---- Tahmin doğruluk takibi: önce geçmiş tahminleri sonuçlandır, sonra yenisini kaydet
resolve_pending_predictions(feat_df)
log_new_prediction(latest.name, latest["close"], future_pred)
sync_logs_to_github()

# ---- Üst metrik satırı --------------------------------------------------
current_price = latest["close"]
change_24h_pct = float(ticker["priceChangePercent"])
volume_24h = float(ticker["volume"])

col1, col2, col3, col4, col5 = st.columns(5)
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

if future_up_prob is not None:
    col3.metric("Sınıflandırıcı: Yukarı Olasılığı", f"%{future_up_prob * 100:.1f}")
else:
    col3.metric("Sınıflandırıcı: Yukarı Olasılığı", "—")

label, reasons = trend_signal(latest)
col4.metric("Kural Tabanlı Trend Sinyali", label)

if metrics:
    col5.metric(
        "Model MAE (Naive'e Karşı)",
        f"${metrics['mae']:,.2f}",
        f"{metrics['edge_vs_naive_pct']:+.1f}% (naive: ${metrics['mae_naive']:,.2f})",
    )
else:
    col5.metric("Model MAE (Naive'e Karşı)", "—")

st.info(f"**Sinyal gerekçeleri:** {reasons}")

if metrics:
    edge = metrics["edge_vs_naive_pct"]
    if edge <= 0:
        st.warning(
            f"⚠️ Model, test setinde naive baseline'ı (tahmin = şu anki fiyat) **geçemiyor** "
            f"(edge: {edge:+.1f}%). Bu ufukta ({HORIZON} dk) modelin fiyat hareketinden bağımsız "
            "bilgi çıkaramadığı anlamına gelir — tahminlere temkinli yaklaşın."
        )
    else:
        st.success(
            f"✅ Model, test setinde naive baseline'ı **{edge:.1f}%** oranında geçiyor. "
            f"Yön isabet oranı: %{metrics['direction_acc'] * 100:.1f} • "
            f"Sınıflandırıcı isabet oranı: %{metrics['clf_test_acc'] * 100:.1f} "
            "(rastgele/coin-flip taban: %50)."
        )

if future_pred is not None:
    target_time = latest.name + pd.Timedelta(minutes=HORIZON)
    render_countdown_and_forecast_widget(target_time, current_price, future_pred, REFRESH_MS, future_up_prob)

st.caption(
    f"Son güncelleme (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} — "
    "Bu uygulama yatırım tavsiyesi değildir, eğitim/analiz amaçlıdır."
)

with st.expander("🔧 Hiperparametre Seçimi (Walk-Forward CV, 30 dk'da bir yenilenir)"):
    st.caption(
        "TimeSeriesSplit (embargo=horizon) ile küçük bir grid taranır; en düşük ortalama "
        "CV-MAE'ye sahip kombinasyon seçilir."
    )
    st.dataframe(cv_report, hide_index=True, use_container_width=True)
    st.write(f"**Seçilen parametreler:** {best_params}")

with st.expander("📚 Eğitim Geçmişi (Hiperparametreler, Hata Oranları, Özellik Önemi) — SQLite'ta kalıcı"):
    st.caption(
        "Her gerçek model eğitiminde (yeni mum verisi geldiğinde) otomatik loglanır — "
        "widget etkileşimlerinde tekrar eğitim yapılmadığı için tekrar loglanmaz. "
        "Bu geçmiş `data` branch'indeki `data/training_log.csv` dosyasına da senkronize edilir."
    )
    runs_df = load_training_runs(limit=20)
    if runs_df.empty:
        st.info("Henüz loglanmış bir eğitim koşusu yok.")
    else:
        display_runs = runs_df.copy()
        display_runs["Zaman (UTC)"] = pd.to_datetime(display_runs["run_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        display_runs["Hiperparametreler"] = display_runs.apply(
            lambda r: f"depth={r['max_depth']}, lr={r['learning_rate']}, n_est={r['n_estimators']}", axis=1
        )
        display_runs["MAE (Naive'e Karşı)"] = display_runs.apply(
            lambda r: f"${r['mae']:,.2f} ({r['edge_vs_naive_pct']:+.1f}% / naive: ${r['mae_naive']:,.2f})", axis=1
        )
        display_runs["Yön İsabeti (Reg / Clf)"] = display_runs.apply(
            lambda r: f"%{r['direction_acc']*100:.1f} / %{r['clf_test_acc']*100:.1f}", axis=1
        )
        display_runs["En Çok Baktığı 5 Özellik"] = display_runs["reg_feature_importance"].apply(
            lambda s: ", ".join(list(json.loads(s).keys())[:5])
        )
        st.dataframe(
            display_runs[[
                "Zaman (UTC)", "Hiperparametreler", "MAE (Naive'e Karşı)",
                "Yön İsabeti (Reg / Clf)", "En Çok Baktığı 5 Özellik",
            ]],
            hide_index=True, use_container_width=True,
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
st.subheader("🎯 Tahmin Doğruluk Takibi (Canlı Backtest, SQLite'ta kalıcı)")
st.caption(
    "Her yenilemede, bir önceki döngüde yapılan tahmin artık gerçekleşen fiyatla "
    "karşılaştırılıp burada gösterilir. Bu geçmiş SQLite'a yazılır; uygulamayı/sunucuyu "
    "yeniden başlatsanız bile kaybolmaz ve zamanla birikir."
)

resolved_df = load_resolved_predictions()

if resolved_df.empty:
    st.warning(
        "Henüz sonuçlanmış bir tahmin yok. İlk tahminin sonucu, tahmin edilen "
        f"{HORIZON} dakikalık süre dolduktan sonraki ilk yenilemede burada görünecek."
    )
else:
    errors_abs = (resolved_df["actual_price"] - resolved_df["predicted_price"]).abs()
    errors_pct = errors_abs / resolved_df["actual_price"] * 100
    pred_dir = np.sign(resolved_df["predicted_price"] - resolved_df["price_at_pred"])
    actual_dir = np.sign(resolved_df["actual_price"] - resolved_df["price_at_pred"])
    direction_hits = ((pred_dir == actual_dir) | (actual_dir == 0)).astype(int)

    acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
    acc_col1.metric("Sonuçlanan Tahmin Sayısı (tüm zamanlar)", len(resolved_df))
    acc_col2.metric("Ortalama Mutlak Hata", f"${errors_abs.mean():,.2f}")
    acc_col3.metric("Ortalama Yüzde Hata", f"{errors_pct.mean():.3f}%")
    acc_col4.metric("Yön İsabet Oranı", f"{direction_hits.mean() * 100:.1f}%")

    display_df = resolved_df.head(30).copy()
    display_df["Tahmin Zamanı (UTC)"] = pd.to_datetime(display_df["pred_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["O Anki Fiyat"] = display_df["price_at_pred"].map(lambda v: f"${v:,.2f}")
    display_df["Tahmin Edilen"] = display_df["predicted_price"].map(lambda v: f"${v:,.2f}")
    display_df["Gerçekleşen"] = display_df["actual_price"].map(lambda v: f"${v:,.2f}")
    display_df["Hata"] = [
        f"${a:,.2f} ({p:.3f}%)" for a, p in zip(errors_abs.head(30), errors_pct.head(30))
    ]
    display_df["Yön Doğru mu?"] = ["✅" if h else "❌" for h in direction_hits.head(30)]

    st.dataframe(
        display_df[["Tahmin Zamanı (UTC)", "O Anki Fiyat", "Tahmin Edilen", "Gerçekleşen", "Hata", "Yön Doğru mu?"]],
        hide_index=True, use_container_width=True,
    )

st.divider()

# ---- Ek indikatörler tablosu ---------------------------------------------
st.subheader("Anlık İndikatör Değerleri")
indicator_table = pd.DataFrame({
    "İndikatör": [
        "EMA 9 / 21 / 50", "SMA 20", "RSI (14)", "MACD / Sinyal",
        "Bollinger Üst / Orta / Alt", "Stokastik %K / %D", "ATR (14)", "VWAP", "OBV",
        "Rejim (15dk) Trend", "Rejim (15dk) RSI",
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
        "Yükseliş" if latest["regime_trend"] > 0 else "Düşüş",
        f"{latest['regime_rsi_14']:.2f}",
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
    "Not: Model her 2 dakikalık yenilemede en güncel pencere üzerinde yeniden eğitilir; "
    "hedef ham fiyat değil log-getiri olarak tanımlanır ve naive baseline ile karşılaştırılır. "
    "Kripto piyasaları yüksek oynaklığa sahiptir; bu araç finansal tavsiye değildir."
)
