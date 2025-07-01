def get_current_features1():
    import requests
    import apimoex
    import pandas as pd
    import logging
    import os
    from copy import deepcopy

    from pandas import DataFrame

    from tinkoff.invest import Client, SecurityTradingStatus
    from tinkoff.invest.services import InstrumentsService
    from tinkoff.invest.utils import quotation_to_decimal

    from datetime import timedelta

    from tinkoff.invest import CandleInterval, Client
    from tinkoff.invest.utils import now

    import warnings

    warnings.filterwarnings("ignore")

    import joblib

    import joblib
    import ta
    import numpy as np
    from copy import deepcopy

    from huggingface_hub import hf_hub_download

    from lightgbm import LGBMRegressor

    # XGBoost
    from xgboost import XGBRegressor

    # CatBoost
    from catboost import CatBoostRegressor

    # Random Forest (из sklearn)
    from sklearn.ensemble import RandomForestRegressor

    TOKEN = "t.YbAt3ov-iNU4jt9A4l9ML4ga77xB1z_NYKOFEvZZDRv72ilghDUEJVk3B86XRSCeyNz5_do2Go_cAqj2qjH9Jg"

    def get_all_quotes_info(figi_series, ticker_series):

        total_quotes = {}

        counter = 0
        for figi, ticker in zip(figi_series, ticker_series):
            if figi != 0:
                with Client(TOKEN) as client:

                    quotes = {
                        "open": [],
                        "high": [],
                        "low": [],
                        "close": [],
                        "volume": [],
                        "datetime": [],
                    }
                    for candle in client.get_all_candles(
                        figi=figi,
                        from_=now() - timedelta(days=30),
                        interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
                    ):

                        open = candle.open.units + candle.open.nano / 1000000000
                        quotes["open"].append(open)

                        high = candle.high.units + candle.high.nano / 1000000000
                        quotes["high"].append(high)

                        low = candle.low.units + candle.low.units / 1000000000
                        quotes["low"].append(low)

                        close = candle.close.units + candle.close.nano / 1000000000
                        quotes["close"].append(close)

                        volume = candle.volume
                        quotes["volume"].append(volume)

                        datetime = candle.time
                        quotes["datetime"].append(datetime)

                    quotes = pd.DataFrame(quotes)

                    total_quotes[ticker] = quotes

            else:
                continue

        return total_quotes

    def get_figi_by_ticker(ticker):

        with Client(TOKEN) as client:
            instruments: InstrumentsService = client.instruments
            tickers = []
            for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
                for item in getattr(instruments, method)().instruments:
                    tickers.append(
                        {
                            "name": item.name,
                            "ticker": item.ticker,
                            "class_code": item.class_code,
                            "figi": item.figi,
                            "uid": item.uid,
                            "type": method,
                            "min_price_increment": quotation_to_decimal(
                                item.min_price_increment
                            ),
                            "scale": 9 - len(str(item.min_price_increment.nano)) + 1,
                            "lot": item.lot,
                            "trading_status": str(
                                SecurityTradingStatus(item.trading_status).name
                            ),
                            "api_trade_available_flag": item.api_trade_available_flag,
                            "currency": item.currency,
                            "exchange": item.exchange,
                            "buy_available_flag": item.buy_available_flag,
                            "sell_available_flag": item.sell_available_flag,
                            "short_enabled_flag": item.short_enabled_flag,
                            "klong": quotation_to_decimal(item.klong),
                            "kshort": quotation_to_decimal(item.kshort),
                        }
                    )

            tickers_df = DataFrame(tickers)

            ticker_df = tickers_df[tickers_df["ticker"] == ticker]
            if ticker_df.empty:

                return (0, 0)

            return ticker_df["figi"].iloc[0], ticker_df.iloc[0]

    # функции
    # а дальше код
    total_features_dict = joblib.load("total_features_dict.joblib")

    print(len(total_features_dict))
    # total_features_dict.remove('Hour')
    # total_features_dict.remove('Day')
    # total_features_dict.remove('Month')
    total_features_dict1 = { i.split('_')[1] : [] for i in total_features_dict }
    for i in total_features_dict:
        total_features_dict1[ i.split('_')[1] ].append(i) 
    del total_features_dict1['std']
    total_features_dict1['RTS'].append(['10min_std_RTS'] )

    # counter = 0 
    # for lst in total_features_dict1.values():
    #     counter += len(lst)
    # print(counter)
    
    # methods_df = {'q75' : [] , 'median' : [] , 'max' : [] , 'obv_' : [] , 'q25' : [] , 'mean' : []  , 'min' : [] , 'std' : [] , 'atr' : []}
    # for ticker , lst in total_features_dict1.items():
    #     for i in lst:
    #         if 'q75' in  i :
    #             methods_df['q75'].append(ticker)
    #         if 'median' in  i :
    #             methods_df['median'].append(ticker)
    #         if 'max' in  i :
    #             methods_df['max'].append(ticker)
    #         if 'obv_' in  i :
    #             methods_df['obv_'].append(ticker)
    #         if 'q25' in  i :
    #             methods_df['q25'].append(ticker)
    #         if 'mean' in  i :
    #             methods_df['mean'].append(ticker)
    #         if 'min' in  i :
    #             methods_df['min'].append(ticker)
    #         if 'std' in  i :
    #             methods_df['std'].append(ticker)
    # methods_df['std'].append('RTS')
    # methods_df['atr'].append('RTS')


    # counter = 0 
    # for lst in methods_df.values():
    #     counter += len(lst)
    #     for i in lst:
    #         if i not in total_features_dict : 
    #             print(i)
    # print(counter)

    from datetime import datetime

    current_year = str(datetime.now().year)[-1]
    current_month = datetime.now().month

    if 1 <= current_month <= 2 or current_month == 12:
        futures_month = "H"
    elif 3 <= current_month <= 5:
        futures_month = "M"
    elif 6 <= current_month <= 8:
        futures_month = "U"
    elif 9 <= current_month <= 11:
        futures_month = "Z"

    month_to_futures_code = {
    1: "F",   # Январь
    2: "G",   # Февраль
    3: "H",   # Март
    4: "J",   # Апрель
    5: "K",   # Май
    6: "M",   # Июнь
    7: "N",   # Июль
    8: "Q",   # Август
    9: "U",   # Сентябрь
    10: "V",  # Октябрь
    11: "X",  # Ноябрь
    12: "Z"   # Декабрь
    }

    features_dict =  { ''}
    total2 = {}
    ticker_dfs = {}
    for ticker , features in total_features_dict1.items():

        ticker1 = deepcopy(ticker)
        print(ticker1)

        if ticker == "IMOEX":
            ticker = "IMOEXF"
            print(ticker)
        if ticker == "RTS":
            ticker = f"RI{futures_month}{current_year}"  # Поменять потом
            print(ticker)
        if ticker == "SI":
            ticker = f"Si{futures_month}{current_year}"
            print(ticker)

        if ticker == "RVI":
            ticker = f"VI{month_to_futures_code[current_month+1]}{current_year}"
            print(ticker)


        ticker_df = get_all_quotes_info(
            [get_figi_by_ticker(ticker)[1]["figi"]], [ticker]
        )[ticker]
        # print(type(ticker_df))
        # print(ticker_df.columns)

        ticker_df = ticker_df.rename(columns  = {'open' : f'open_{ticker1}' , 'close' : f'close_{ticker1}' , 'low' : f'low_{ticker1}' , 'high' : f'high_{ticker1}' , 'volume' : f'volume_{ticker1}' } ) 
        
        ticker_df = ticker_df.set_index('datetime')
        ticker_df[f'ln_ret_close_{ticker1}'] = np.log( ticker_df[f'close_{ticker1}'] / ticker_df[f'close_{ticker1}'].shift(1)) * 100
        ticker_df[f'ln_ret_high_{ticker1}'] = np.log(ticker_df[f'high_{ticker1}']  / ticker_df[f'high_{ticker1}'].shift(1)) * 100
        ticker_df[f'ln_ret_open_{ticker1}'] = np.log(ticker_df[f'open_{ticker1}']  / ticker_df[f'open_{ticker1}'].shift(1)) * 100
        ticker_df[f'ln_ret_low_{ticker1}'] = np.log(ticker_df[f'low_{ticker1}'] / ticker_df[f'low_{ticker1}'].shift(1)) * 100
        ticker_df[f'ln_ret_volume_{ticker1}'] = np.log(ticker_df[f'volume_{ticker1}']  / ticker_df[f'volume_{ticker1}'].shift(1)) * 100
        
        lst = total_features_dict1[ticker1] 

        s = ticker_df[f'close_{ticker1}']
        roll = s.rolling(window=15) 
        from ta.volume import AccDistIndexIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator, NegativeVolumeIndexIndicator, OnBalanceVolumeIndicator, VolumePriceTrendIndicator, VolumeWeightedAveragePrice
        from ta.volatility import AverageTrueRange, BollingerBands
        for method in lst:
            if 'q75' in method:
                
                
                ticker_df[f"close_{ticker1}_q75_15"]      = roll.quantile(0.75)
            if 'median' in method:
                ticker_df[f"close_{ticker1}_median_15"] = roll.median()
            if 'max' in method:
                ticker_df[f"close_{ticker1}_max_15"] = roll.max()
            if 'obv_' in method:
                obv = OnBalanceVolumeIndicator(
                    close= ticker_df[f'ln_ret_close_{ticker1}'],
                volume= ticker_df[f'ln_ret_volume_{ticker1}']
                )

                ticker_df[f'obv_{ticker1}'] = obv.on_balance_volume()
            if 'q25' in method:  
                ticker_df[f"close_{ticker1}_q25_15"]      = roll.quantile(0.25)
            if 'mean' in method:
                ticker_df[f"close_{ticker1}_mean_15"]      = roll.mean()
            if 'min' in method:
                ticker_df[f"close_{ticker1}_min_15"]      = roll.min()
            if 'atr' in method:
                atr = AverageTrueRange(
                    high=ticker_df[f'ln_ret_high_{ticker1}'],
                low= ticker_df[f'ln_ret_low_{ticker1}'],
                close = ticker_df[f'ln_ret_close_{ticker1}'],
                window = 5 ,
                
                fillna= True )

                ticker_df[f'atr_{ticker1}'] = atr.average_true_range() 
            if 'std' in method:
                ticker_df[f'10min_std_{ticker1}'] = roll.std() 
        

#         {'10min_std_RTS',
#  'atr_RTS',
#  'close_NVTK_count_above_mean_15',
#  'close_RVI_mean_change_15',
#  'close_RVI_skew_15',
#  'close_RVI_var_15'}
        if ticker1 == 'NVTK' : 
            ticker_df['close_NVTK_mean_15'] = roll.mean()
            ticker_df['close_NVTK_count_above_mean_15'] =  roll.apply(lambda x: np.sum(x > x.mean()), raw=True)

        if ticker1 == 'RVI':
            ticker_df['close_RVI_mean_change_15'] = roll.mean()
            ticker_df['close_RVI_skew_15'] = roll.skew()
            ticker_df['close_RVI_var_15'] = roll.std()
        
        if ticker1 == 'RTS':
            ticker_df['10min_std_RTS'] = roll.std()


        ticker_df.drop(columns = [f'open_{ticker1}' , f'high_{ticker1}' , f'low_{ticker1}' , f'close_{ticker1}' , f'volume_{ticker1}'] , axis = 1 , inplace = True )
        ticker_df.drop(columns = [f'ln_ret_open_{ticker1}' , f'ln_ret_high_{ticker1}' , f'ln_ret_low_{ticker1}' , f'ln_ret_close_{ticker1}' , f'ln_ret_volume_{ticker1}'] , axis = 1 , inplace = True )
        ticker_df = ticker_df.reset_index()
        ticker_df.drop( columns = ['datetime'] , axis = 1 , inplace  = True  )
       
        ticker_df = ticker_df.iloc[-30 : ]
        ticker_df.index = pd.Index([i for i in range(30)])
        ticker_dfs[ticker1] = ticker_df
        

        print(ticker_dfs[ticker1])
        
    combined_df = pd.concat(ticker_dfs.values(), axis= 1 )
    now = datetime.now()
    day = now.day
    month = now.month
    hour = now.hour
    # combined_df['Month'] = [ month for i in range(len(combined_df))]
    # combined_df['Day'] = [ day for i in range(len(combined_df))]
    # combined_df['Hour'] = [ hour for i in range(len(combined_df))]

    
    combined_df['Day'] =    [now.day for i in range( len(combined_df) ) ] 
    combined_df['Month'] =    [now.month for i in range( len(combined_df) ) ] 
    combined_df['Hour'] =    [now.hour for i in range( len(combined_df) ) ] 

    print(combined_df.shape)

    import joblib
    joblib.dump(combined_df , 'combined_df.joblib')
    for col in combined_df.columns : 
        if col not in total_features_dict : 
            print(col)
    
    lst = joblib.load('total_features.joblib')
    total_df  = combined_df[lst]
    
    print(total_df)
    print(total_df.shape)

    MODEL_PATH = "randomforest_model.joblib"
    REPO_ID = "IvanBorrow/MLFG"
    FILENAME = "randomforest_model.joblib"

    if not os.path.exists(MODEL_PATH):
        print("📥 Скачиваем модель с Hugging Face...")
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=".", local_dir_use_symlinks=False)
        print('Скачан')

    catboost_model = joblib.load('catboost_model.joblib')
    lgbm_model = joblib.load('lgbm_model.joblib')
    xgb_model = joblib.load('xgb_model.joblib')
    randomforest_model = joblib.load('randomforest_model.joblib')

    catboost_preds = catboost_model.predict(total_df)
    print(catboost_preds)
    lgbm_preds = lgbm_model.predict(total_df)
    print(lgbm_preds)
    xgb_preds = xgb_model.predict(total_df)
    print(xgb_preds)
    randomforest_preds = randomforest_model.predict(total_df)
    print(randomforest_preds)

    meta_X = df = pd.DataFrame({
    'xgb_pred': xgb_preds,
    'cb_pred': catboost_preds,
    'lgbm_pred': lgbm_preds,
    'rf_pred':  randomforest_preds ,
    })

    print(meta_X)

    from itertools import combinations
    def build_meta_features(meta_X_raw: pd.DataFrame, win: int = 10, lags: list = [1, 2, 3]) -> pd.DataFrame:
        df   = meta_X_raw.copy()
        cols = df.columns                           # базовые прогнозы ────



        # 1. pair-wise diff / ratio
        for c1, c2 in combinations(cols, 2):
            df[f"diff_{c1}_{c2}"]  = df[c1] - df[c2]
            df[f"ratio_{c1}_{c2}"] = df[c1] / (df[c2].replace(0, np.nan))  # Inf → NaN

        # 2. ранги
        ranks = meta_X_raw.rank(axis=1, method="dense")
        ranks.columns = [f"rank_{c}" for c in cols]
        df = pd.concat([df, ranks], axis=1)

        # 3. rolling-фичи
        roll = meta_X_raw.rolling(win, min_periods=win)
        roll_mean  = roll.mean().add_suffix(f"_mean_{win}")
        roll_std   = roll.std().add_suffix(f"_std_{win}")
        roll_min   = roll.min().add_suffix(f"_min_{win}")
        roll_max   = roll.max().add_suffix(f"_max_{win}")
    
        roll_stats = pd.concat([roll_mean, roll_std, roll_min, roll_max], axis=1)

        # 4. объединяем и удаляем строки, где ВСЕ rolling-колонки = NaN
        mask_has_any = roll_stats.notna().any(axis=1)
        out = pd.concat([df, roll_stats], axis=1)[mask_has_any].astype("float32")

        return out
    
    meta_X_plus = build_meta_features(meta_X, win=10, lags=[1, 2, 3])
    print('meta_X_plus' , meta_X_plus)

    MODEL_PATH = "randomforest_metamodel.joblib"
    REPO_ID = "IvanBorrow/MLFG"
    FILENAME = "randomforest_metamodel.joblib"

    if not os.path.exists(MODEL_PATH):
        print("📥 Скачиваем Meta модель с Hugging Face...")
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=".", local_dir_use_symlinks=False)
        print('Скачан')
    
    randomforest_metamodel = joblib.load('randomforest_metamodel.joblib')

    total_preds = randomforest_metamodel.predict(meta_X_plus)
    print('total_preds' , total_preds)
    total_pred  = total_preds[-1]
    print(total_pred)

    return total_pred

if __name__ == "__main__":
    df = get_current_features1()
    print(df)
