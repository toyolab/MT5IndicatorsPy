# coding: utf-8
import numpy as np
import pandas as pd
from scipy.signal import lfilter
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

# 共通移動平均 on Array
def MAonArray(a, ma_period, ma_method):
    if ma_method == 'SMA':
        h = np.ones(ma_period)/ma_period
        y = lfilter(h, 1, a)
        y[:ma_period-1] = 'NaN'
    elif ma_method == 'EMA':
        alpha = 2/(ma_period+1)
        y,zf = lfilter([alpha], [1,alpha-1], a, zi=[a[0]*(1-alpha)])
    elif ma_method == 'SMMA':
        alpha = 1/ma_period
        y,zf = lfilter([alpha], [1,alpha-1], a, zi=[a[0]*(1-alpha)])
    elif ma_method == 'LWMA':
        h = np.arange(ma_period, 0, -1)*2/ma_period/(ma_period+1)
        y = lfilter(h, 1, a)
        y[:ma_period-1] = 'NaN'
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
@jit
def iCCI(df, ma_period, applied_price='Typical'):
    SP = MAonArray(df[applied_price], ma_period, 'SMA')
    price = df[applied_price].values
    M = price - SP
    D = np.zeros(len(M))
    for i in range(len(D)):
        for j in range(ma_period):
            D[i] += np.abs(price[i-j] - SP[i])
    D *= 0.015/ma_period
    return pd.Series(M/D, index=df.index)

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

# iAMA()関数
@jit
def iAMA(df, ma_period, fast_period, slow_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    Signal = price.diff(ma_period).abs()
    Noise = price.diff().abs().rolling(ma_period).sum()
    ER = Signal.values/Noise.values
    FastSC = 2/(fast_period+1)
    SlowSC = 2/(slow_period+1)
    SSC = ER*(FastSC-SlowSC)+SlowSC
    price = price.values
    AMA = price.copy()
    for i in range(ma_period, len(AMA)):
        AMA[i] = AMA[i-1] + SSC[i]*SSC[i]*(price[i]-AMA[i-1])
    return pd.Series(AMA, index=df.index).shift(ma_shift)

# iFrAMA()関数
@jit
def iFrAMA(df, ma_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    H = df['High']
    L = df['Low']
    N1 = (H.rolling(ma_period).max()-L.rolling(ma_period).min())/ma_period 
    N2 = (H.shift(ma_period).rolling(ma_period).max()-L.shift(ma_period).rolling(ma_period).min())/ma_period 
    N3 = (H.rolling(2*ma_period).max()-L.rolling(2*ma_period).min())/(2*ma_period)
    D = (np.log(N1.values+N2.values)-np.log(N3.values))/np.log(2)
    A = np.exp(-4.6*(D-1))
    price = price.values
    FRAMA = price.copy()
    for i in range(2*ma_period, len(FRAMA)):
        FRAMA[i] = FRAMA[i-1] + A[i]*(price[i]-FRAMA[i-1])
    return pd.Series(FRAMA, index=df.index).shift(ma_shift)

# iRVI()関数
def iRVI(df, ma_period):
    CO = df['Close'].values - df['Open'].values
    HL = df['High'].values - df['Low'].values
    MA = lfilter([1/6,1/3,1/3,1/6], 1, CO)
    RA = lfilter([1/6,1/3,1/3,1/6], 1, HL)
    Main = MAonArray(MA, ma_period, 'SMA') / MAonArray(RA, ma_period, 'SMA')
    Signal = lfilter([1/6,1/3,1/3,1/6], 1, Main)
    return pd.DataFrame({'Main': Main, 'Signal': Signal},
                        columns=['Main', 'Signal'], index=df.index)

# iWPR()関数
def iWPR(df, period):
    Max = df['High'].rolling(period).max()
    Min = df['Low'].rolling(period).min()
    return (df['Close']-Max)/(Max-Min)*100

# iVIDyA()関数
@jit
def iVIDyA(df, cmo_period, ma_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    UpSum = price.diff().clip_lower(0).rolling(cmo_period).sum()
    DnSum = -price.diff().clip_upper(0).rolling(cmo_period).sum()
    CMO = np.abs((UpSum-DnSum)/(UpSum+DnSum)).values
    price = price.values
    VIDYA = price.copy()
    for i in range(cmo_period, len(VIDYA)):
        VIDYA[i] = VIDYA[i-1] + 2/(ma_period+1)*CMO[i]*(price[i]-VIDYA[i-1])
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
    Hline = high.rolling(Kperiod).max()
    Lline = low.rolling(Kperiod).min()
    sumlow = (df['Close']-Lline).rolling(slowing).sum()
    sumhigh = (Hline-Lline).rolling(slowing).sum()
    Main = sumlow/sumhigh*100
    Signal = MAonSeries(Main, Dperiod, ma_method)
    return pd.DataFrame({'Main': Main, 'Signal': Signal},
                        columns=['Main', 'Signal'])

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
    last_period = 0
    dir_long = True
    ACC = step
    SAR = df['Close'].values.copy()
    High = df['High'].values
    Low = df['Low'].values
    for i in range(1,len(SAR)):
        last_period += 1
        if dir_long == True:
            Ep1 = High[i-last_period:i].max()
            SAR[i] = SAR[i-1]+ACC*(Ep1-SAR[i-1])
            Ep0 = max([Ep1, High[i]])
            if Ep0 > Ep1 and ACC+step <= maximum: ACC+=step
            if SAR[i] > Low[i]:
                dir_long = False
                SAR[i] = Ep0
                last_period = 0
                ACC = step
        else:
            Ep1 = Low[i-last_period:i].min()
            SAR[i] = SAR[i-1]+ACC*(Ep1-SAR[i-1])
            Ep0 = min([Ep1, Low[i]])
            if Ep0 < Ep1 and ACC+step <= maximum: ACC+=step
            if SAR[i] < High[i]:
                dir_long = True
                SAR[i] = Ep0
                last_period = 0
                ACC = step
    return pd.Series(SAR, index=df.index)

# 各関数のテスト
if __name__ == '__main__':

    file = 'USDJPY.f16385.txt'
    ohlc = pd.read_csv(file, index_col='Time', parse_dates=True)
    ohlc_ext = ext_ohlc(ohlc)

    #x = iMA(ohlc_ext, 14, ma_method='SMMA', applied_price='Close')
    #x = iATR(ohlc_ext, 14)
    #x = iDEMA(ohlc_ext, 14, applied_price='Close')
    #x = iTEMA(ohlc_ext, 14, applied_price='Close')
    #x = iMomentum(ohlc_ext, 14)
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
    x = iRVI(ohlc_ext, 10)
    #x = iWPR(ohlc_ext, 14)
    #x = iVIDyA(ohlc_ext, 15, 12)
    #x = iBands(ohlc_ext, 20, 2, bands_shift=5)
    #x = iStochastic(ohlc_ext, 10, 3, 5, ma_method='LWMA', price_field='CLOSECLOSE')
    #x = iHLBand(ohlc, 20)
    #x = iAlligator(ohlc_ext, 13, 8, 8, 5, 5, 3)
    #x = iGator(ohlc_ext, 13, 8, 8, 5, 5, 3)
    #x = iADX(ohlc_ext, 14)
    #x = iADXWilder(ohlc_ext, 14)
    #x = iSAR(ohlc_ext, 0.02, 0.2)

    #diff = ohlc['Ind0'] - x
    diff0 = ohlc['Ind0'] - x['Main']
    diff1 = ohlc['Ind1'] - x['Signal']
    #diff1 = ohlc['Ind1'] - x['PlusDI']
    #diff2 = ohlc['Ind2'] - x['MinusDI']
