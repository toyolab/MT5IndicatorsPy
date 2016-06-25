# MT5IndicatorsPy
MT5 compatible technical indicator functions in Python

MetaTrader5のテクニカル指標関数をPythonで記述したものです。関数の仕様やテクニカル指標の説明については、テクニカル指標関数一覧を参照してください。`OutputRates.mq5`は、各関数のテストデータを出力するMetaTrader5のスクリプトです。

すべての関数に共通なのは、`symbol`と`period`の二つのパラメータが`DataFrame`クラスの`df`というパラメータに置き換わったところです。`df`は、`'Open','High','Low','Close','Median','Typical','Weighted'`というラベルの付いた`Series`クラスの時系列データから構成されます。

`ma_method`には、`'SMA','EMA','SMMA','LWMA'`のいずれかを代入します。

`applied_price`には、`'Close','Open','High','Low','Median','Typical','Weighted'`のいずれかを代入します。

`price_field`には、`'LOWHIGH','CLOSECLOSE'`のいずれかを代入します。

出力が一つの関数は`Series`クラスの時系列データ、二つ以上の関数は`DataFrame`クラスの時系列データとして出力され、出力の種類は指定したラベルで区別します。

## テクニカル指標関数一覧
* [iAC()](https://www.mql5.com/ja/docs/indicators/iac) - [ACオシレーター](http://www.metatrader5.com/ja/terminal/help/indicators/bw_indicators/ao)
*    iAD not implemented, requires volume
* [iADX()](https://www.mql5.com/ja/docs/indicators/iadx) - [ADX](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/admi) #`'Main','PlusDI','MinusDI'`
* [iADXWilder()](https://www.mql5.com/ja/docs/indicators/iadxwilder) - [ワイルダーのADX](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/admiw) #`'Main','PlusDI','MinusDI'`
* [iAlligator()](https://www.mql5.com/ja/docs/indicators/ialligator) - [アリゲーター](http://www.metatrader5.com/ja/terminal/help/indicators/bw_indicators/alligator) #`'Jaw','Teeth','Lips'`
* [iAMA()](https://www.mql5.com/ja/docs/indicators/iama) - [適応型移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/ama)
* [iAO()](https://www.mql5.com/ja/docs/indicators/iao) - [オーサムオシレーター](http://www.metatrader5.com/ja/terminal/help/indicators/bw_indicators/awesome)
* [iATR()](https://www.mql5.com/ja/docs/indicators/iatr) - [ATR](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/atr) #`ma_method`追加
* [iBearsPower()](https://www.mql5.com/ja/docs/indicators/ibearspower) - [ベアパワー](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/bears)
* [iBands()](https://www.mql5.com/ja/docs/indicators/ibands) - [ボリンジャーバンド](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/bb) #`'Base','Upper','Lower'`
* [iBullsPower()](https://www.mql5.com/ja/docs/indicators/ibullspower) - [ブルパワー](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/bulls)
* [iCCI()](https://www.mql5.com/ja/docs/indicators/icci) - [CCI](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/cci)
*    iChaikin not implemented, requires volume
*    iCustom not implemented
* [iDEMA()](https://www.mql5.com/ja/docs/indicators/idema) - [2重指数移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/dema)
* [iDeMarker()](https://www.mql5.com/ja/docs/indicators/idemarker) - [デマーカー](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/demarker)
* [iEnvelopes()](https://www.mql5.com/ja/docs/indicators/ienvelopes) - [エンベローブ](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/envelopes) #`'Upper','Lower'`
*    iForce not implemented, requires volume
*    iFractals not implemented
* [iFrAMA()](https://www.mql5.com/ja/docs/indicators/iframa) - [フラクタル適応型移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/fama)
* [iGator()](https://www.mql5.com/ja/docs/indicators/igator) - [ゲーターオシレーター](http://www.metatrader5.com/ja/terminal/help/indicators/bw_indicators/go) #`'Upper','Lower'`
* iHLBand() not included in MQL5 - HLバンド #`'Upper','Lower'`
*    iIchimoku not implemented
*    iBWMFI not implemented, requires volume
* [iMomentum()](https://www.mql5.com/ja/docs/indicators/imomentum) - [モメンタム](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/momentum)
*    iMFI not implemented, requires volume
* [iMA()](https://www.mql5.com/ja/docs/indicators/ima) - [移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/ma)
* [iOsMA()](https://www.mql5.com/ja/docs/indicators/iosma) - [移動平均オシレーター](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/mao)
* [iMACD()](https://www.mql5.com/ja/docs/indicators/imacd) - [MACD](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/macd) #`'Main','Signal'`
*    iOBV not implemented, requires volume
* [iSAR()](https://www.mql5.com/ja/docs/indicators/isar) - [パラボリック SAR](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/psar)
* [iRSI()](https://www.mql5.com/ja/docs/indicators/irsi) - [RSI](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/rsi)
* [iRVI()](https://www.mql5.com/ja/docs/indicators/irvi) - [相対活力指数](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/rvi) #`'Main','Signal'`
* [iStdDev()](https://www.mql5.com/ja/docs/indicators/istddev) - [標準偏差](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/sd) #`ma_method`削除
* [iStochastic()](https://www.mql5.com/ja/docs/indicators/istochastic) - [ストキャスティックス](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/so) #`'Main','Signal'`
* [iTEMA()](https://www.mql5.com/ja/docs/indicators/itema) - [3重指数移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/tema)
* [iTriX()](https://www.mql5.com/ja/docs/indicators/itrix) - [3重指数移動平均オシレーター](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/tea)
* [iWPR()](https://www.mql5.com/ja/docs/indicators/iwpr) - [ウィリアムパーセントレンジ](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/wpr)
* [iVIDyA()](https://www.mql5.com/ja/docs/indicators/ividya) - [可変インデックス動的平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/vida)
*    iVolumes not implemented, requires volume
