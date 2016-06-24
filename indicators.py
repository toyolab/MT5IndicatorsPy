# coding: utf-8
import numpy as np
import pandas as pd

# dfのデータからtfで指定するタイムフレームの4本足データを作成する関数
def TF_ohlc(df, tf):
    x = df.resample(tf).ohlc()
    O = x['Open']['open']
    H = x['High']['high']
    L = x['Low']['low']
    C = x['Close']['close']
    ret = pd.DataFrame({'Open': O, 'High': H, 'Low': L, 'Close': C},
                       columns=['Open','High','Low','Close'])
    return ret.dropna()

# dfのデータに Median, Typical, Weighted price を追加する関数
def ext_ohlc(df):
    O = df['Open']
    H = df['High']
    L = df['Low']
    C = df['Close']
    ext = pd.DataFrame({'Median': (H+L)/2,
                       'Typical': (H+L+C)/3,
                       'Weighted': (H+L+C*2)/4},
                       columns=['Median','Typical','Weighted'])
    return pd.concat([df,ext], axis=1)

# 共通移動平均
def MAonSeries(s, ma_period, ma_method):
    if ma_method == 'SMA':    
        return s.rolling(ma_period).mean()
    elif ma_method == 'EMA':
        return s.ewm(span=ma_period).mean()
    elif ma_method == 'SMMA':
        return s.ewm(alpha=1/ma_period).mean()
    elif ma_method == 'LWMA':
        y = pd.Series(0.0, index=s.index)
        for i in range(len(y)):
            if i<ma_period-1: y[i] = 'NaN'
            else:
                y[i] = 0
                for j in range(ma_period):
                    y[i] += s[i-j]*(ma_period-j)
                y[i] /= ma_period*(ma_period+1)/2
        return y
    
# iMA()関数
def iMA(df, ma_period, ma_shift=0, ma_method='SMA', applied_price='Close'):
    return MAonSeries(df[applied_price], ma_period, ma_method).shift(ma_shift)

# iATR()関数
def iATR(df, ma_period, ma_method='SMA'):
    TR = pd.DataFrame({'H':df['High'], 'C':df['Close'].shift()}).max(1)\
       - pd.DataFrame({'L':df['Low'], 'C':df['Close'].shift()}).min(1)
    return MAonSeries(TR, ma_period, ma_method)
    
# iDEMA()関数
def iDEMA(df, ma_period, ma_shift=0, applied_price='Close'):
    EMA = MAonSeries(df[applied_price], ma_period, ma_method='EMA')
    EMA2 = MAonSeries(EMA, ma_period, ma_method='EMA')
    return (2*EMA-EMA2).shift(ma_shift)

# iTEMA()関数
def iTEMA(df, ma_period, ma_shift=0, applied_price='Close'):
    EMA = MAonSeries(df[applied_price], ma_period, ma_method='EMA')
    EMA2 = MAonSeries(EMA, ma_period, ma_method='EMA')
    EMA3 = MAonSeries(EMA2, ma_period, ma_method='EMA')
    return (3*EMA-3*EMA2+EMA3).shift(ma_shift)

# iMomentum()関数
def iMomentum(df, mom_period, applied_price='Close'):
    price = df[applied_price]
    shift = price.shift(mom_period)
    return price/shift*100
    
# iRSI()関数
def iRSI(df, ma_period, applied_price='Close'):
    diff = df[applied_price].diff()
    positive = MAonSeries(diff.clip_lower(0), ma_period, 'SMMA')
    negative = MAonSeries(diff.clip_upper(0), ma_period, 'SMMA')
    return 100-100/(1-positive/negative)
    
# iStdDev()関数
def iStdDev(df, ma_period, ma_shift=0, applied_price='Close'):
    return df[applied_price].rolling(ma_period).std(ddof=0).shift(ma_shift)

# iAO()関数
def iAO(df):
    return MAonSeries(df['Median'], 5, 'SMA') - MAonSeries(df['Median'], 34, 'SMA')

# iAC()関数
def iAC(df):
    AO = iAO(df)
    return AO - MAonSeries(AO, 5, 'SMA')

# iBearsPower()関数
def iBearsPower(df, ma_period):
    return df['Low'] - MAonSeries(df['Close'], ma_period, 'EMA')

# iBullsPower()関数
def iBullsPower(df, ma_period):
    return df['High'] - MAonSeries(df['Close'], ma_period, 'EMA')

# iCCI()関数
def iCCI(df, ma_period, applied_price='Typical'):
    price = df[applied_price]
    SP = MAonSeries(price, ma_period, 'SMA')
    M = price - SP
    D = pd.Series(0.0, index=df.index)
    for i in range(len(D)):
        for j in range(ma_period):
            D[i] += np.abs(price[i-j] - SP[i])
    D *= 0.015/ma_period
    return M/D

# iDeMarker()関数
def iDeMarker(df, ma_period):
    DeMax = df['High'].diff().clip_lower(0)
    DeMin = -df['Low'].diff().clip_upper(0)
    SDeMax = MAonSeries(DeMax, ma_period, 'SMA')
    SDeMin = MAonSeries(DeMin, ma_period, 'SMA')
    return SDeMax/(SDeMax+SDeMin)

# iEnvelopes()関数
def iEnvelopes(df, ma_period, deviation, ma_shift=0, ma_method='SMA', applied_price='Close'):
    price = df[applied_price]
    MA = MAonSeries(price, ma_period, ma_method).shift(ma_shift)
    Upper = MA*(1+deviation/100)
    Lower = MA*(1-deviation/100)
    return pd.DataFrame({'Upper': Upper, 'Lower': Lower},
                        columns=['Upper', 'Lower'])

# iMACD()関数
def iMACD(df, fast_period, slow_period, signal_period, applied_price='Close'):
    price = df[applied_price]
    Main = MAonSeries(price, fast_period, 'EMA') - MAonSeries(price, slow_period, 'EMA')
    Signal = MAonSeries(Main, signal_period, 'SMA')
    return pd.DataFrame({'Main': Main, 'Signal': Signal},
                        columns=['Main', 'Signal'])

# iOsMA()関数
def iOsMA(df, fast_period, slow_period, signal_period, applied_price='Close'):
    MACD = iMACD(df, fast_period, slow_period, signal_period, applied_price)
    return MACD['Main'] - MACD['Signal']

# iTriX()関数
def iTriX(df, ma_period, applied_price='Close'):
    EMA1 = MAonSeries(df[applied_price], ma_period, 'EMA')
    EMA2 = MAonSeries(EMA1, ma_period, 'EMA')
    EMA3 = MAonSeries(EMA2, ma_period, 'EMA')
    return EMA3.diff()/EMA3.shift()

# 各関数のテスト
if __name__ == '__main__':

    file = 'USDJPY.f16385.txt'
    ohlc = pd.read_csv(file, index_col='Time', parse_dates=True)
    ohlc_ext = ext_ohlc(ohlc)

    #x = iMA(ohlc, 14, ma_shift=0, ma_method='EMA', applied_price='Open')
    #x = iATR(ohlc, 14)
    #x = iDEMA(ohlc, 14, ma_shift=0, applied_price='Close')
    #x = iTEMA(ohlc, 14, ma_shift=0, applied_price='Close')
    #x = iMomentum(ohlc, 14)
    #x = iRSI(ohlc, 14)
    #x = iStdDev(ohlc_ext, 14, ma_shift=3, applied_price='Weighted')
    #x = iAO(ohlc_ext)
    #x = iAC(ohlc_ext)
    #x = iBearsPower(ohlc_ext, 13)
    #x = iBullsPower(ohlc_ext, 13)
    #x = iCCI(ohlc_ext, 14)
    #x = iDeMarker(ohlc_ext, 14)
    #x = iEnvelopes(ohlc_ext, 10, 1)
    #x = iMACD(ohlc_ext, 12, 26, 9)
    #x = iOsMA(ohlc_ext, 12, 26, 9)
    x = iTriX(ohlc_ext, 14)

    diff = ohlc['Ind0'] - x
    #diff0 = ohlc['Ind0'] - x['Main']
    #diff1 = ohlc['Ind1'] - x['Signal']
