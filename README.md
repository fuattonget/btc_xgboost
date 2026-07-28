# BTC Trend Analizi & XGBoost Fiyat Tahmin Uygulaması

Yerelde (localhost) çalışan, Binance'den canlı BTC/USDT verisi çeken, popüler trade
indikatörlerini hesaplayan ve XGBoost ile **2 dakika sonrasının** fiyatını tahmin eden
bir Streamlit web uygulaması.

## Özellikler

- **Canlı veri**: Binance Public REST API (`/api/v3/klines`) — API key gerekmez.
- **İndikatörler**: EMA (9/21/50), SMA(20), RSI(14), MACD, Bollinger Bantları,
  Stokastik Osilatör, ATR(14), OBV, VWAP.
- **Model**: XGBoost regresyon, lag özellikleri + indikatörlerle 2 dakika sonraki
  kapanış fiyatını tahmin eder. Model her yenilemede en güncel 1000 mumluk pencere
  üzerinde yeniden eğitilir; test seti üzerinde MAE/RMSE gösterilir.
- **Kural tabanlı trend sinyali**: EMA dizilimi, MACD, RSI, Bollinger ve Stokastik'i
  birleştiren basit bir "Güçlü Yükseliş / Nötr / Güçlü Düşüş" özeti.
- **Otomatik yenileme**: Sayfa her 2 dakikada bir kendini yeniler, yeni veriyi çeker
  ve modeli yeniden eğitir.

## Kurulum

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacaktır.

## Notlar / Sınırlamalar

- Bu uygulama **yatırım tavsiyesi değildir**; eğitim ve analiz amaçlıdır.
- Binance API'ye internet erişimi gereklidir. Bazı ülkelerde/ağlarda Binance
  API'sine erişim kısıtlı olabilir; böyle bir durumda VPN gerekebilir ya da
  `app.py` içindeki `BINANCE_KLINES_URL` başka bir borsanın (örn. Bybit, Kraken)
  eşdeğer public endpoint'iyle değiştirilebilir.
- Model her yenilemede sıfırdan eğitildiği için ilk açılış birkaç saniye sürebilir.
- 1 dakikalık mumlarla çalışıldığı için "2 dakika sonrası tahmin", modelin 2 bar
  ileriye baktığı anlamına gelir (`HORIZON = 2` değişkeniyle `app.py` içinden
  ayarlanabilir).
- İndikatör periyotlarını, tahmin ufkunu (`HORIZON`) veya çekilen mum sayısını
  (`LOOKBACK`) `app.py` dosyasının en üstündeki sabitlerden değiştirebilirsiniz.
