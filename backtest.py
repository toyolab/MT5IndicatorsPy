# coding: utf-8
import numpy as np
import pandas as pd

# バックテスト
def Backtest(ohlc, BuyEntry, SellEntry, BuyExit, SellExit, lots=0.1, spread=2):
    Open = ohlc['Open'].values #始値
    Point = 0.0001 #1pipの値
    if(Open[0] > 50): Point = 0.01 #クロス円の1pipの値
    Spread = spread*Point #スプレッド
    Lots = lots*100000 #実際の売買量
    N = len(ohlc) #FXデータのサイズ
    BuyExit[N-2] = SellExit[N-2] = True #最後に強制エグジット
    BuyPrice = SellPrice = 0.0 # 売買価格
    
    LongTrade = np.zeros(N) # 買いトレード情報
    ShortTrade = np.zeros(N) # 売りトレード情報
    
    LongPL = np.zeros(N) # 買いポジションの損益
    ShortPL = np.zeros(N) # 売りポジションの損益

    for i in range(1,N):
        if BuyEntry[i-1] and BuyPrice == 0: #買いエントリーシグナル
            BuyPrice = Open[i]+Spread
            LongTrade[i] = BuyPrice #買いポジションオープン
        elif BuyExit[i-1] and BuyPrice != 0: #買いエグジットシグナル
            ClosePrice = Open[i]
            LongTrade[i] = -ClosePrice #買いポジションクローズ
            LongPL[i] = (ClosePrice-BuyPrice)*Lots #損益確定
            BuyPrice = 0

        if SellEntry[i-1] and SellPrice == 0: #売りエントリーシグナル
            SellPrice = Open[i]
            ShortTrade[i] = SellPrice #売りポジションオープン
        elif SellExit[i-1] and SellPrice != 0: #売りエグジットシグナル
            ClosePrice = Open[i]+Spread
            ShortTrade[i] = -ClosePrice #売りポジションクローズ
            ShortPL[i] = (SellPrice-ClosePrice)*Lots #損益確定
            SellPrice = 0

    return pd.DataFrame({'Long':LongTrade, 'Short':ShortTrade}, index=ohlc.index),\
            pd.DataFrame({'Long':LongPL, 'Short':ShortPL}, index=ohlc.index)

# バックテストレポート
def BacktestReport(Trade, PL):
    LongPL = PL['Long']
    ShortPL = PL['Short']
    LongTrades = np.count_nonzero(Trade['Long'])//2
    ShortTrades = np.count_nonzero(Trade['Short'])//2
    GrossProfit = LongPL.clip_lower(0).sum()+ShortPL.clip_lower(0).sum()
    GrossLoss = LongPL.clip_upper(0).sum()+ShortPL.clip_upper(0).sum()
    #総損益
    Profit = GrossProfit+GrossLoss
    #取引数
    Trades = LongTrades+ShortTrades
    #平均損益
    if Trades==0: Average = 0
    else: Average = Profit/Trades
    #プロフィットファクター
    if GrossLoss==0: PF=0
    else: PF = -GrossProfit/GrossLoss
    #最大ドローダウン
    Equity = (LongPL+ShortPL).cumsum()
    MDD = (Equity.cummax()-Equity).max()
    #リカバリーファクター
    if MDD==0: RF=0
    else: RF = Profit/MDD
    return np.array([Profit, Trades, Average, PF, MDD, RF])

# 各関数のテスト
if __name__ == '__main__':
	pass
