# MT5IndicatorsPy
MT5 compatible technical indicator functions in Python

MetaTrader5のテクニカル指標関数をPythonで記述したものです。関数の仕様やテクニカル指標の説明については、テクニカル指標関数一覧を参照してください。

すべての関数に共通なのは、`symbol`と`period`の二つのパラメータが`DataFrame`クラスの`df`というパラメータに置き換わったところです。`df`は、`'Open','High','Low','Close','Median','Typical','Weighted'`というラベルの付いた`Series`クラスの時系列データを含みます。

`ma_method`には、`'SMA','EMA','SMMA','LWMA'`のいずれかを代入します。

`applied_price`には、`'Close','Open','High','Low','Median','Typical','Weighted'`のいずれかを代入します。

## テクニカル指標関数一覧
* [iAC()](https://www.mql5.com/ja/docs/indicators/iac) - [ACオシレーター](http://www.metatrader5.com/ja/terminal/help/indicators/bw_indicators/ao)
*    iAD not implemented, requires volume
*    iADX
*    iADXWilder
*    iAlligator
*    iAMA
* [iAO()](https://www.mql5.com/ja/docs/indicators/iao) - [オーサムオシレータ](http://www.metatrader5.com/ja/terminal/help/indicators/bw_indicators/awesome)
* [iATR()](https://www.mql5.com/ja/docs/indicators/iatr) - [ATR](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/atr) #`ma_method`追加
* [iBearsPower()](https://www.mql5.com/ja/docs/indicators/ibearspower) - [ベアパワー](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/bears)
*    iBands
* [iBullsPower()](https://www.mql5.com/ja/docs/indicators/ibullspower) - [ブルパワー](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/bulls)
* [iCCI()](https://www.mql5.com/ja/docs/indicators/icci) - [商品チャンネル指数](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/cci)
*    iChaikin not implemented, requires volume
*    iCustom not implemented
* [iDEMA()](https://www.mql5.com/ja/docs/indicators/idema) - [2重指数移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/dema)
*    iDeMarker
*    iEnvelopes
*    iForce
*    iFractals
*    iFrAMA
*    iGator
*    iIchimoku
*    iBWMFI
* [iMomentum()](https://www.mql5.com/ja/docs/indicators/imomentum) - [モメンタム](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/momentum)
*    iMFI not implemented, requires volume
* [iMA()](https://www.mql5.com/ja/docs/indicators/ima) - [移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/ma)
*    iOsMA
*    iMACD
*    iOBV not implemented, requires volume
*    iSAR
* [iRSI()](https://www.mql5.com/ja/docs/indicators/irsi) - [相対力指数](http://www.metatrader5.com/ja/terminal/help/indicators/oscillators/rsi)
*    iRVI
* [iStdDev()](https://www.mql5.com/ja/docs/indicators/istddev) - [標準偏差](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/sd) #`ma_method`削除
*    iStochastic
* [iTEMA()](https://www.mql5.com/ja/docs/indicators/itema) - [3重指数移動平均](http://www.metatrader5.com/ja/terminal/help/indicators/trend_indicators/tema)
*    iTriX
*    iWPR
*    iVIDyA
*    iVolumes not implemented, requires volume
