//+------------------------------------------------------------------+
//|                                                  OutputRates.mq5 |
//+------------------------------------------------------------------+
#property script_show_inputs

input datetime start_time = D'01.01.2015';
input datetime stop_time = D'01.01.2016';

void OnStart()
{
   int hFile = FileOpen(_Symbol+IntegerToString(_Period)+".txt", FILE_WRITE|FILE_CSV|FILE_ANSI);
   if(hFile == INVALID_HANDLE) return;
   FileWrite(hFile, "Time,Open,High,Low,Close,Ind"); 

   //int hInd = iMA(_Symbol, 0, 14, 0, MODE_EMA, PRICE_CLOSE);
   //int hInd = iATR(_Symbol, 0, 14);
   //int hInd = iDEMA(_Symbol, 0, 14, 0, PRICE_CLOSE);
   int hInd = iTEMA(_Symbol, 0, 14, 0, PRICE_CLOSE);

   MqlRates rates[];
   double buf[];
   int copied = CopyRates(Symbol(), 0, start_time, stop_time, rates);
   int icopied = CopyBuffer(hInd, 0, start_time, stop_time, buf);
   if(icopied != copied) MessageBox("copy error");
   for(int i=0; i<copied; i++)
   {
      string out = TimeToString(rates[i].time);
      string format = ",%G,%G,%G,%G,%.15G";
      out = out+" "+StringFormat(format,
                                  rates[i].open,
                                  rates[i].high,
                                  rates[i].low,
                                  rates[i].close,
                                  buf[i]);
      FileWrite(hFile, out); 
   }
   FileClose(hFile);
   MessageBox(IntegerToString(copied)+" data written");
}
