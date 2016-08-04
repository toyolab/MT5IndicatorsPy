# coding: utf-8
import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi
from numba import jit

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

#シフト関数
def shift(x, n=1):
    return np.concatenate((np.zeros(n), x[:-n]))

# SMA on Array
@jit
def SMAonArray(x, ma_period):
    x[np.isnan(x)] = 0
    y = np.empty_like(x)
    y[:ma_period-1] = np.nan
    y[ma_period-1] = np.sum(x[:ma_period])
    for i in range(ma_period, len(x)):
        y[i] = y[i-1] + x[i] - x[i-ma_period]
    return y/ma_period

# EMA on Array
@jit
def EMAonArray(x, alpha):
    x[np.isnan(x)] = 0
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1,len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

# Adaptive EMA on Array
@jit
def AEMAonArray(x, alpha):
    x[np.isnan(x)] = 0
    alpha[np.isnan(alpha)] = 0
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1,len(x)):
        y[i] = alpha[i]*x[i] + (1-alpha[i])*y[i-1]
    return y

# 共通移動平均 on Array
def MAonArray(a, ma_period, ma_method):
    if ma_method == 'SMA':
        y = SMAonArray(a, ma_period)
    elif ma_method == 'EMA':
        y = EMAonArray(a, 2/(ma_period+1))
    elif ma_method == 'SMMA':
        y = EMAonArray(a, 1/ma_period)
    elif ma_method == 'LWMA':
        h = np.arange(ma_period, 0, -1)*2/ma_period/(ma_period+1)
        y = lfilter(h, 1, a)
        y[:ma_period-1] = np.nan
    return y
    
# 共通移動平均 on Series
def MAonSeries(s, ma_period, ma_method):
    return pd.Series(MAonArray(s.values, ma_period, ma_method), index=s.index)
    
# iMA()関数
def iMA(df, ma_period, ma_shift=0, ma_method='SMA', applied_price='Close'):
    return MAonSeries(df[applied_price], ma_period, ma_method).shift(ma_shift)

# iATR()関数
def iATR(df, ma_period, ma_method='SMA'):
    TR = np.max(np.vstack((df['High'].values, shift(df['Close'].values))).T, axis=1)\
       - np.min(np.vstack((df['Low'].values, shift(df['Close'].values))).T, axis=1)
    return pd.Series(MAonArray(TR, ma_period, ma_method), index=df.index)
    
# iDEMA()関数
@jit
def iDEMA(df, ma_period, ma_shift=0, applied_price='Close'):
    alpha = 2/(ma_period+1)
    a1 = 2*(alpha-1)
    a2 = (1-alpha)**2
    b0 = alpha*(2-alpha)
    b1 = 2*alpha*(alpha-1)
    x = df[applied_price].values
    y = np.empty_like(x)
    y[0] = x[0]
    y[1] = b0*x[1] + b1*x[0] - a1*y[0] - a2*y[0]
    for i in range(2,len(x)):
        y[i] = b0*x[i] + b1*x[i-1] - a1*y[i-1] - a2*y[i-2]
    return pd.Series(y, index=df.index).shift(ma_shift)

# iTEMA()関数
@jit
def iTEMA(df, ma_period, ma_shift=0, applied_price='Close'):
    alpha = 2/(ma_period+1)
    a1 = 3*(alpha-1)
    a2 = 3*(1-alpha)**2
    a3 = (alpha-1)**3
    b0 = 3*alpha*(1-alpha)+alpha**3
    b1 = 3*alpha*(alpha-2)*(1-alpha)
    b2 = 3*alpha*(1-alpha)**2
    x = df[applied_price].values
    y = np.empty_like(x)
    y[0] = x[0]
    y[1] = b0*x[1] + b1*x[0] + b2*x[0] - a1*y[0] - a2*y[0] - a3*y[0]
    y[2] = b0*x[2] + b1*x[1] + b2*x[0] - a1*y[1] - a2*y[0] - a3*y[0]
    for i in range(3,len(x)):
        y[i] = b0*x[i] + b1*x[i-1] + b2*x[i-2] - a1*y[i-1] - a2*y[i-2] - a3*y[i-3]
    return pd.Series(y, index=df.index).shift(ma_shift)

# iMomentum()関数
@jit
def iMomentum(df, mom_period, applied_price='Close'):
    x = df[applied_price].values
    y = np.empty_like(x)
    y[:mom_period] = np.nan
    for i in range(mom_period, len(x)):
        y[i] = x[i]/x[i-mom_period]*100
    return pd.Series(y, index=df.index)
    
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
    AO = iAO(df).values
    return pd.Series(AO - SMAonArray(AO, 5), index=df.index)

# iBearsPower()関数
def iBearsPower(df, ma_period):
    return df['Low'] - MAonSeries(df['Close'], ma_period, 'EMA')

# iBullsPower()関数
def iBullsPower(df, ma_period):
    return df['High'] - MAonSeries(df['Close'], ma_period, 'EMA')

# iCCI()関数
@jit
def iCCI(df, ma_period, applied_price='Typical'):
    price = df[applied_price].values
    SP = SMAonArray(price, ma_period)
    M = price - SP
    D = np.zeros(len(M))
    for i in range(len(D)):
        for j in range(ma_period):
            D[i] += abs(price[i-j] - SP[i])
    D *= 0.015/ma_period
    return pd.Series(M/D, index=df.index)

# iDeMarker()関数
def iDeMarker(df, ma_period):
    DeMax = df['High'].diff().clip_lower(0).values
    DeMin = -df['Low'].diff().clip_upper(0).values
    SDeMax = SMAonArray(DeMax, ma_period)
    SDeMin = SMAonArray(DeMin, ma_period)
    return pd.Series(SDeMax/(SDeMax+SDeMin), index=df.index)

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
    price = df[applied_price].values
    Main = MAonArray(price, fast_period, 'EMA') - MAonArray(price, slow_period, 'EMA')
    Signal = SMAonArray(Main, signal_period)
    return pd.DataFrame({'Main': Main, 'Signal': Signal},
                        columns=['Main', 'Signal'], index=df.index)

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

# iAMA()関数
def iAMA(df, ma_period, fast_period, slow_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    Signal = price.diff(ma_period).abs()
    Noise = price.diff().abs().rolling(ma_period).sum()
    ER = Signal.values/Noise.values
    FastSC = 2/(fast_period+1)
    SlowSC = 2/(slow_period+1)
    SSC = ER*(FastSC-SlowSC)+SlowSC
    AMA = AEMAonArray(price.values, SSC*SSC)
    return pd.Series(AMA, index=df.index).shift(ma_shift)

# iFrAMA()関数
def iFrAMA(df, ma_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    H = df['High']
    L = df['Low']
    N1 = (H.rolling(ma_period).max()-L.rolling(ma_period).min())/ma_period 
    N2 = (H.shift(ma_period).rolling(ma_period).max()-L.shift(ma_period).rolling(ma_period).min())/ma_period 
    N3 = (H.rolling(2*ma_period).max()-L.rolling(2*ma_period).min())/(2*ma_period)
    D = (np.log(N1.values+N2.values)-np.log(N3.values))/np.log(2)
    A = np.exp(-4.6*(D-1))
    FRAMA = AEMAonArray(price.values, A)
    return pd.Series(FRAMA, index=df.index).shift(ma_shift)

# iRVI()関数
def iRVI(df, ma_period):
    CO = df['Close'].values - df['Open'].values
    HL = df['High'].values - df['Low'].values
    MA = lfilter([1/6,1/3,1/3,1/6], 1, CO)
    RA = lfilter([1/6,1/3,1/3,1/6], 1, HL)
    Main = SMAonArray(MA, ma_period) / SMAonArray(RA, ma_period)
    Signal = lfilter([1/6,1/3,1/3,1/6], 1, Main)
    return pd.DataFrame({'Main': Main, 'Signal': Signal},
                        columns=['Main', 'Signal'], index=df.index)

# iWPR()関数
def iWPR(df, period):
    Max = df['High'].rolling(period).max()
    Min = df['Low'].rolling(period).min()
    return (df['Close']-Max)/(Max-Min)*100

# iVIDyA()関数
def iVIDyA(df, cmo_period, ma_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    UpSum = price.diff().clip_lower(0).rolling(cmo_period).sum()
    DnSum = -price.diff().clip_upper(0).rolling(cmo_period).sum()
    CMO = np.abs((UpSum-DnSum)/(UpSum+DnSum)).values
    VIDYA = AEMAonArray(price.values, 2/(ma_period+1)*CMO)
    return pd.Series(VIDYA, index=df.index).shift(ma_shift)

# iBands()関数
def iBands(df, bands_period, deviation, bands_shift=0, applied_price='Close'):
    price = df[applied_price].shift(bands_shift)
    Base = price.rolling(bands_period).mean()
    sigma = price.rolling(bands_period).std(ddof=0)
    Upper = Base+sigma*deviation
    Lower = Base-sigma*deviation
    return pd.DataFrame({'Base': Base, 'Upper': Upper, 'Lower': Lower},
                        columns=['Base', 'Upper', 'Lower'])

# iStochastic()関数
def iStochastic(df, Kperiod, Dperiod, slowing, ma_method='SMA', price_field='LOWHIGH'):
    if price_field == 'LOWHIGH':
        high = df['High']
        low = df['Low']
    elif price_field == 'CLOSECLOSE':
        high = low = df['Close']
    Hline = high.rolling(Kperiod).max().values
    Lline = low.rolling(Kperiod).min().values
    close = df['Close'].values
    sumlow = SMAonArray(close-Lline, slowing)
    sumhigh = SMAonArray(Hline-Lline, slowing)
    Main = sumlow/sumhigh*100
    Signal = MAonArray(Main, Dperiod, ma_method)
    return pd.DataFrame({'Main': Main, 'Signal': Signal},
                        columns=['Main', 'Signal'], index=df.index)

# iHLBand()関数
def iHLBand(df, band_period, band_shift=0, price_field='LOWHIGH'):
    if price_field == 'LOWHIGH':
        high = df['High']
        low = df['Low']
    elif price_field == 'CLOSECLOSE':
        high = low = df['Close']
    Upper = high.rolling(band_period).max().shift(band_shift)
    Lower = low.rolling(band_period).min().shift(band_shift)
    return pd.DataFrame({'Upper': Upper, 'Lower': Lower},
                        columns=['Upper', 'Lower'])

# iAlligator()関数
def iAlligator(df, jaw_period, jaw_shift, teeth_period, teeth_shift,
               lips_period, lips_shift, ma_method='SMMA', applied_price='Median'):
    price = df[applied_price]
    Jaw = MAonSeries(price, jaw_period, ma_method).shift(jaw_shift)
    Teeth = MAonSeries(price, teeth_period, ma_method).shift(teeth_shift)
    Lips = MAonSeries(price, lips_period, ma_method).shift(lips_shift)
    return pd.DataFrame({'Jaw': Jaw, 'Teeth': Teeth, 'Lips': Lips},
                        columns=['Jaw', 'Teeth', 'Lips'])

# iGator()関数
def iGator(df, jaw_period, jaw_shift, teeth_period, teeth_shift,
               lips_period, lips_shift, ma_method='SMMA', applied_price='Median'):
    AG = iAlligator(df, jaw_period, jaw_shift, teeth_period, teeth_shift,
                    lips_period, lips_shift, ma_method, applied_price)
    Upper = (AG['Jaw']-AG['Teeth']).abs()
    Lower = -(AG['Teeth']-AG['Lips']).abs()
    return pd.DataFrame({'Upper': Upper, 'Lower': Lower},
                        columns=['Upper', 'Lower'])

# iADX()関数
def iADX(df, adx_period):
    dP = df['High'].diff().clip_lower(0).values
    dM = -df['Low'].diff().clip_upper(0).values
    dM[dP > dM] = 0
    dP[dP < dM] = 0
    dP[0] = dP[1]
    dM[0] = dM[1]
    TR = np.max(np.vstack((df['High'].values, shift(df['Close'].values))).T, axis=1)\
       - np.min(np.vstack((df['Low'].values, shift(df['Close'].values))).T, axis=1)
    PlusDI = 100*MAonArray(dP/TR, adx_period, 'EMA')
    MinusDI = 100*MAonArray(dM/TR, adx_period, 'EMA')
    Main = MAonArray(100*np.abs(PlusDI-MinusDI)/(PlusDI+MinusDI), adx_period, 'EMA')
    return pd.DataFrame({'Main': Main, 'PlusDI': PlusDI, 'MinusDI': MinusDI},
                        columns=['Main', 'PlusDI', 'MinusDI'], index=df.index)

# iADXWilder()関数
def iADXWilder(df, adx_period):
    dP = df['High'].diff().clip_lower(0).values
    dM = -df['Low'].diff().clip_upper(0).values
    dM[dP > dM] = 0
    dP[dP < dM] = 0
    dP[0] = dP[1]
    dM[0] = dM[1]
    ATR = iATR(df, adx_period, 'SMMA').values
    PlusDI = 100*MAonArray(dP, adx_period, 'SMMA')/ATR
    MinusDI = 100*MAonArray(dM, adx_period, 'SMMA')/ATR
    Main = MAonArray(100*np.abs(PlusDI-MinusDI)/(PlusDI+MinusDI), adx_period, 'SMMA')
    return pd.DataFrame({'Main': Main, 'PlusDI': PlusDI, 'MinusDI': MinusDI},
                        columns=['Main', 'PlusDI', 'MinusDI'], index=df.index)

# iSAR()関数
@jit
def iSAR(df, step, maximum):
    dir_long = True
    ACC = step
    SAR = df['Close'].values.copy()
    High = df['High'].values
    Low = df['Low'].values
    Ep1 = High[0]
    for i in range(1,len(SAR)):
        if dir_long == True:
            Ep1 = max(Ep1, High[i-1])
            SAR[i] = SAR[i-1]+ACC*(Ep1-SAR[i-1])
            if High[i] > Ep1: ACC = min(ACC+step, maximum)
            if SAR[i] > Low[i]:
                dir_long = False
                SAR[i] = Ep1
                ACC = step
                Ep1 = Low[i]
        else:
            Ep1 = min(Ep1, Low[i-1])
            SAR[i] = SAR[i-1]+ACC*(Ep1-SAR[i-1])
            if Low[i] < Ep1: ACC = min(ACC+step, maximum)
            if SAR[i] < High[i]:
                dir_long = True
                SAR[i] = Ep1
                ACC = step
                Ep1 = High[i]
    return pd.Series(SAR, index=df.index)

# 各関数のテスト
if __name__ == '__main__':

    file = 'USDJPY.f16385.txt'
    ohlc = pd.read_csv(file, index_col='Time', parse_dates=True)
    ohlc_ext = ext_ohlc(ohlc)

    #x = iMA(ohlc_ext, 14, ma_method='SMA')
    #x = iMA(ohlc_ext, 14, ma_method='EMA')
    #x = iMA(ohlc_ext, 14, ma_method='SMMA', applied_price='Median')
    #x = iMA(ohlc_ext, 14, ma_method='LWMA', applied_price='Typical')
    #x = iATR(ohlc_ext, 14)
    #x = iDEMA(ohlc_ext, 14, applied_price='Close')
    #x = iTEMA(ohlc_ext, 14, applied_price='Close')
    x = iMomentum(ohlc_ext, 14)
    #x = iRSI(ohlc_ext, 14)
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
    #x = iTriX(ohlc_ext, 14)
    #x = iAMA(ohlc_ext, 15, 2, 30)
    #x = iFrAMA(ohlc_ext, 14)
    #x = iRVI(ohlc_ext, 10)
    #x = iWPR(ohlc_ext, 14)
    #x = iVIDyA(ohlc_ext, 15, 12)
    #x = iBands(ohlc_ext, 20, 2, bands_shift=5)
    #x = iStochastic(ohlc_ext, 10, 3, 5, ma_method='SMA', price_field='LOWHIGH')
    #x = iHLBand(ohlc, 20)
    #x = iAlligator(ohlc_ext, 13, 8, 8, 5, 5, 3)
    #x = iGator(ohlc_ext, 13, 8, 8, 5, 5, 3)
    #x = iADX(ohlc_ext, 14)
    #x = iADXWilder(ohlc_ext, 14)
    #x = iSAR(ohlc_ext, 0.02, 0.2)

    dif = ohlc['Ind0'] - x
    #dif0 = ohlc['Ind0'] - x['Main']
    #dif1 = ohlc['Ind1'] - x['Signal']
    #dif1 = ohlc['Ind1'] - x['PlusDI']
    #dif2 = ohlc['Ind2'] - x['MinusDI']
