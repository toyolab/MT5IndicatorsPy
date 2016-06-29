# coding: utf-8
import numpy as np
import pandas as pd
from scipy.signal import lfilter

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
        h = np.ones(ma_period)/ma_period
        y = lfilter(h, 1, s)
        y[:ma_period-1] = 'NaN'
    elif ma_method == 'EMA':
        alpha = 2/(ma_period+1)
        y,zf = lfilter([alpha], [1,alpha-1], s, zi=[s[0]*(1-alpha)])
    elif ma_method == 'SMMA':
        alpha = 1/ma_period
        y,zf = lfilter([alpha], [1,alpha-1], s, zi=[s[0]*(1-alpha)])
    elif ma_method == 'LWMA':
        h = np.arange(ma_period, 0, -1)*2/ma_period/(ma_period+1)
        y = lfilter(h, 1, s)
        y[:ma_period-1] = 'NaN'
    return pd.Series(y, index=s.index)
    
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

# iAMA()関数
def iAMA(df, ma_period, fast_period, slow_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    Signal = price.diff(ma_period).abs()
    Noise = price.diff().abs().rolling(ma_period).sum()
    ER = Signal/Noise
    FastSC = 2/(fast_period+1)
    SlowSC = 2/(slow_period+1)
    SSC = ER*(FastSC-SlowSC)+SlowSC
    AMA = price.copy()
    for i in range(ma_period, len(AMA)):
        AMA[i] = AMA[i-1] + SSC[i]*SSC[i]*(price[i]-AMA[i-1])
    return AMA.shift(ma_shift)

# iFrAMA()関数
def iFrAMA(df, ma_period, ma_shift=0, applied_price='Close'):
    price = df[applied_price]
    H = df['High']
    L = df['Low']
    N1 = (H.rolling(ma_period).max()-L.rolling(ma_period).min())/ma_period 
    N2 = (H.shift(ma_period).rolling(ma_period).max()-L.shift(ma_period).rolling(ma_period).min())/ma_period 
    N3 = (H.rolling(2*ma_period).max()-L.rolling(2*ma_period).min())/(2*ma_period)
    D = (np.log(N1+N2)-np.log(N3))/np.log(2)
    A = np.exp(-4.6*(D-1))
    FRAMA = price.copy()
    for i in range(2*ma_period, len(FRAMA)):
        FRAMA[i] = FRAMA[i-1] + A[i]*(price[i]-FRAMA[i-1])
    return FRAMA.shift(ma_shift)

# iRVI()関数
def iRVI(df, ma_period):
    CO = df['Close']-df['Open']
    HL = df['High']-df['Low']
    MA = CO+2*(CO.shift(1)+CO.shift(2))+CO.shift(3)
    RA = HL+2*(HL.shift(1)+HL.shift(2))+HL.shift(3)
    Main = MA.rolling(ma_period).sum()/RA.rolling(ma_period).sum()
    Signal = (Main+2*(Main.shift(1)+Main.shift(2))+Main.shift(3))/6
    return pd.DataFrame({'Main': Main, 'Signal': Signal},
                        columns=['Main', 'Signal'])

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
    CMO = (UpSum-DnSum)/(UpSum+DnSum)
    VIDYA = price.copy()
    for i in range(cmo_period, len(VIDYA)):
        VIDYA[i] = VIDYA[i-1] + 2/(ma_period+1)*np.abs(CMO[i])*(price[i]-VIDYA[i-1])
    return VIDYA.shift(ma_shift)

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
    dP = df['High'].diff().clip_lower(0)
    dM = -df['Low'].diff().clip_upper(0)
    for i in range(len(dP)):
        if dP[i] > dM[i]: dM[i] = 0
        if dP[i] < dM[i]: dP[i] = 0
    TR = pd.DataFrame({'H':df['High'], 'C':df['Close'].shift()}).max(1)\
       - pd.DataFrame({'L':df['Low'], 'C':df['Close'].shift()}).min(1)
    PlusDI = 100*MAonSeries(dP/TR, adx_period, 'EMA')
    MinusDI = 100*MAonSeries(dM/TR, adx_period, 'EMA')
    Main = MAonSeries(100*(PlusDI-MinusDI).abs()/(PlusDI+MinusDI), adx_period, 'EMA')
    return pd.DataFrame({'Main': Main, 'PlusDI': PlusDI, 'MinusDI': MinusDI},
                        columns=['Main', 'PlusDI', 'MinusDI'])

# iADXWilder()関数
def iADXWilder(df, adx_period):
    dP = df['High'].diff().clip_lower(0)
    dM = -df['Low'].diff().clip_upper(0)
    for i in range(len(dP)):
        if dP[i] > dM[i]: dM[i] = 0
        if dP[i] < dM[i]: dP[i] = 0
    ATR = iATR(df, adx_period, 'SMMA')
    PlusDI = 100*MAonSeries(dP, adx_period, 'SMMA')/ATR
    MinusDI = 100*MAonSeries(dM, adx_period, 'SMMA')/ATR
    Main = MAonSeries(100*(PlusDI-MinusDI).abs()/(PlusDI+MinusDI), adx_period, 'SMMA')
    return pd.DataFrame({'Main': Main, 'PlusDI': PlusDI, 'MinusDI': MinusDI},
                        columns=['Main', 'PlusDI', 'MinusDI'])

# iSAR()関数
def iSAR(df, step, maximum):
    last_period = 0
    dir_long = True
    ACC = step
    SAR = df['Close'].copy()
    for i in range(1,len(df)):
        last_period += 1    
        if dir_long == True:
            Ep1 = df['High'][i-last_period:i].max()
            SAR[i] = SAR[i-1]+ACC*(Ep1-SAR[i-1])
            Ep0 = max([Ep1, df['High'][i]])
            if Ep0 > Ep1 and ACC+step <= maximum: ACC+=step
            if SAR[i] > df['Low'][i]:
                dir_long = False
                SAR[i] = Ep0
                last_period = 0
                ACC = step
        else:
            Ep1 = df['Low'][i-last_period:i].min()
            SAR[i] = SAR[i-1]+ACC*(Ep1-SAR[i-1])
            Ep0 = min([Ep1, df['Low'][i]])
            if Ep0 < Ep1 and ACC+step <= maximum: ACC+=step
            if SAR[i] < df['High'][i]:
                dir_long = True
                SAR[i] = Ep0
                last_period = 0
                ACC = step
    return SAR

# 各関数のテスト
if __name__ == '__main__':

    file = 'USDJPY.f16385.txt'
    ohlc = pd.read_csv(file, index_col='Time', parse_dates=True)
    ohlc_ext = ext_ohlc(ohlc)

    x = iMA(ohlc, 14, ma_shift=0, ma_method='SMMA', applied_price='Close')
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
    #x = iTriX(ohlc_ext, 14)
    #x = iAMA(ohlc_ext, 15, 2, 30)
    #x = iFrAMA(ohlc_ext, 14)
    #x = iRVI(ohlc_ext, 10)
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

    diff = ohlc['Ind0'] - x
    #diff0 = ohlc['Ind0'] - x['Main']
    #diff1 = ohlc['Ind1'] - x['PlusDI']
    #diff2 = ohlc['Ind2'] - x['MinusDI']
