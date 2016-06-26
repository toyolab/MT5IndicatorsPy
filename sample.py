# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import indicators as ind #テクニカル指標関数のインポート

file = 'FXsample.csv'
ohlc = pd.read_csv(file, index_col='Time', parse_dates=True)
ohlc_ext = ind.ext_ohlc(ohlc) #価格データの追加

#アリゲーター
AG = ind.iAlligator(ohlc_ext, 13, 8, 8, 5, 5, 3)

pd.DataFrame({'Close': ohlc['Close'], 'Jaw': AG['Jaw'], 'Teeth': AG['Teeth'], 'Lips': AG['Lips']},
            columns=['Close', 'Jaw', 'Teeth', 'Lips']).plot(figsize=(8,6),style=[':','-','-','-'])
plt.show()
