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
    cols = joblib.load("lst2.joblib")

    ohlcv = [
        "open_SI",
        "high_SI",
        "low_SI",
        "close_SI",
        "vol_SI",
        "open_RTS",
        "high_RTS",
        "low_RTS",
        "close_RTS",
        "vol_RTS",
        "open_IMOEX",
        "high_IMOEX",
        "low_IMOEX",
        "close_IMOEX",
        "vol_IMOEX",
        "open_GOLD",
        "high_GOLD",
        "low_GOLD",
        "close_GOLD",
        "vol_GOLD",
        "open_AFLT",
        "high_AFLT",
        "low_AFLT",
        "close_AFLT",
        "volume_AFLT",
        "open_CHMF",
        "high_CHMF",
        "low_CHMF",
        "close_CHMF",
        "volume_CHMF",
        "open_FEES",
        "high_FEES",
        "low_FEES",
        "close_FEES",
        "volume_FEES",
        "open_GAZP",
        "high_GAZP",
        "low_GAZP",
        "close_GAZP",
        "volume_GAZP",
        "open_GMKN",
        "high_GMKN",
        "low_GMKN",
        "close_GMKN",
        "volume_GMKN",
        "open_HYDR",
        "high_HYDR",
        "low_HYDR",
        "close_HYDR",
        "volume_HYDR",
        "open_IRAO",
        "high_IRAO",
        "low_IRAO",
        "close_IRAO",
        "volume_IRAO",
        "open_LKOH",
        "high_LKOH",
        "low_LKOH",
        "close_LKOH",
        "volume_LKOH",
        "open_MAGN",
        "high_MAGN",
        "low_MAGN",
        "close_MAGN",
        "volume_MAGN",
        "open_MTSS",
        "high_MTSS",
        "low_MTSS",
        "close_MTSS",
        "volume_MTSS",
        "open_NLMK",
        "high_NLMK",
        "low_NLMK",
        "close_NLMK",
        "volume_NLMK",
        "open_NVTK",
        "high_NVTK",
        "low_NVTK",
        "close_NVTK",
        "volume_NVTK",
        "open_ROSN",
        "high_ROSN",
        "low_ROSN",
        "close_ROSN",
        "volume_ROSN",
        "open_RTKM",
        "high_RTKM",
        "low_RTKM",
        "close_RTKM",
        "volume_RTKM",
        "open_SBERP",
        "high_SBERP",
        "low_SBERP",
        "close_SBERP",
        "volume_SBERP",
        "open_SNGSP",
        "high_SNGSP",
        "low_SNGSP",
        "close_SNGSP",
        "volume_SNGSP",
        "open_SNGS",
        "high_SNGS",
        "low_SNGS",
        "close_SNGS",
        "volume_SNGS",
        "open_TATN",
        "high_TATN",
        "low_TATN",
        "close_TATN",
        "volume_TATN",
        "open_VTBR",
        "high_VTBR",
        "low_VTBR",
        "close_VTBR",
        "volume_VTBR",
        "open_AFKS_M1",
        "high_AFKS_M1",
        "low_AFKS_M1",
        "close_AFKS_M1",
        "volume_AFKS_M1",
        "open_ALRS_M1",
        "high_ALRS_M1",
        "low_ALRS_M1",
        "close_ALRS_M1",
        "volume_ALRS_M1",
        "open_MOEX_M1",
        "high_MOEX_M1",
        "low_MOEX_M1",
        "close_MOEX_M1",
        "volume_MOEX_M1",
        "open_MTLR_M1",
        "high_MTLR_M1",
        "low_MTLR_M1",
        "close_MTLR_M1",
        "volume_MTLR_M1",
    ]

    tickers = [
        "SI",
        "RTS",
        "IMOEX",
        "GOLD",
        "AFLT",
        "CHMF",
        "FEES",
        "GAZP",
        "GMKN",
        "HYDR",
        "IRAO",
        "LKOH",
        "MAGN",
        "MTSS",
        "NLMK",
        "NVTK",
        "ROSN",
        "RTKM",
        "SBERP",
        "SNGSP",
        "SNGS",
        "TATN",
        "VTBR",
        "AFKS",
        "ALRS",
        "MOEX",
        "MTLR",
    ]

    close_lst = [
        "close_SI",
        "close_RVI",
        "close_RTS",
        "close_IMOEX",
        "close_GOLD",
        "close_AFLT",
        "close_CHMF",
        "close_FEES",
        "close_GAZP",
        "close_GMKN",
        "close_HYDR",
        "close_IRAO",
        "close_LKOH",
        "close_MAGN",
        "close_MTSS",
        "close_NLMK",
        "close_NVTK",
        "close_ROSN",
        "close_RTKM",
        "close_SBERP",
        "close_SNGSP",
        "close_SNGS",
        "close_TATN",
        "close_VTBR",
        "close_AFKS_M1",
        "close_ALRS_M1",
        "close_MOEX_M1",
        "close_MTLR_M1",
    ]

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

    total2 = {}
    ticker_dfs = {}
    for ticker in tickers:

        ticker1 = deepcopy(ticker)
        if ticker in ["AFKS", "ALRS", "MOEX", "MTLR"]:
            ticker1 = ticker1 + "_M1"

        if ticker == "IMOEX":
            ticker = "IMOEXF"
        if ticker == "RTS":
            ticker = f"RI{futures_month}{current_year}"  # Поменять потом
            print(ticker)
        if ticker == "SI":
            ticker = f"Si{futures_month}{current_year}"
            print(ticker)

        ticker_df = get_all_quotes_info(
            [get_figi_by_ticker(ticker)[1]["figi"]], [ticker]
        )[ticker]
        print(ticker, ticker_df)
        ticker_df.rename(
            columns={
                "open": f"open_{ticker1}",
                "high": f"high_{ticker1}",
                "low": f"low_{ticker1}",
                "close": f"close_{ticker1}",
                "volume": f"vol_{ticker1}",
            },
            inplace=True,
        )
        ticker_df = ticker_df[
            [
                i
                for i in ticker_df.columns
                if "open" in i
                or "close" in i
                or "low" in i
                or "high" in i
                or "vol" in i
            ]
        ]

        for i in ticker_df.columns:
            ticker_df[i] = (
                np.log(ticker_df[i].astype(float) / ticker_df[i].shift(1).astype(float))
                * 100
            )

        ticker_df = ticker_df.dropna()

        ticker_df[f"sma_5_close_{ticker1}"] = (
            ticker_df[f"close_{ticker1}"].rolling(window=5).mean()
        )
        ticker_df[f"ema_5_close_{ticker1}"] = (
            ticker_df[f"close_{ticker1}"].ewm(span=5).mean()
        )

        if ticker1 not in ["AFKS", "ALRS", "MOEX", "MTLR"]:
            ticker_df[f"close_minus_open_{ticker1}"] = (
                ticker_df[f"close_{ticker1}"] - ticker_df[f"open_{ticker1}"]
            )

        else:
            ticker_df[f"""close_minus_open_{ticker1.split('_')[0]}"""] = (
                ticker_df[f"close_{ticker1}"] - ticker_df[f"open_{ticker1}"]
            )

        ticker_df[f"5min_std_{ticker1}"] = ticker_df[f"close_{ticker1}"].rolling(
            window=5
        ).std() * np.sqrt(5)
        ticker_df[f"10min_std_{ticker1}"] = ticker_df[f"close_{ticker1}"].rolling(
            window=10
        ).std() * np.sqrt(10)

        ticker_df[f"down_streak_5min_{ticker1}"] = (
            ticker_df[f"close_{ticker1}"].lt(0).astype(int).rolling(5).sum()
        )
        ticker_df[f"down_streak_10min_{ticker1}"] = (
            ticker_df[f"close_{ticker1}"].lt(0).astype(int).rolling(10).sum()
        )

        from ta.volume import MFIIndicator

        mfi5 = MFIIndicator(
            high=ticker_df[f"high_{ticker1}"],
            low=ticker_df[f"low_{ticker1}"],
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            window=5,
            fillna=True,
        )

        mfi10 = MFIIndicator(
            high=ticker_df[f"high_{ticker1}"],
            low=ticker_df[f"low_{ticker1}"],
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            window=10,
            fillna=True,
        )

        ticker_df[f"mfi5_{ticker1}"] = mfi5.money_flow_index()
        ticker_df[f"mfi10_{ticker1}"] = mfi10.money_flow_index()

        from ta.volume import (
            AccDistIndexIndicator,
            ChaikinMoneyFlowIndicator,
            EaseOfMovementIndicator,
            ForceIndexIndicator,
            NegativeVolumeIndexIndicator,
            OnBalanceVolumeIndicator,
            VolumePriceTrendIndicator,
            VolumeWeightedAveragePrice,
        )

        acc = AccDistIndexIndicator(
            high=ticker_df[f"high_{ticker1}"],
            low=ticker_df[f"low_{ticker1}"],
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            fillna=True,
        )

        cmf = ChaikinMoneyFlowIndicator(
            high=ticker_df[f"high_{ticker1}"],
            low=ticker_df[f"low_{ticker1}"],
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            window=10,
            fillna=True,
        )

        eom = EaseOfMovementIndicator(
            high=ticker_df[f"high_{ticker1}"],
            low=ticker_df[f"low_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            window=10,
            fillna=True,
        )

        fii = ForceIndexIndicator(
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            window=10,
            fillna=True,
        )

        try:
            nvi = NegativeVolumeIndexIndicator(
                close=ticker_df[f"close_{ticker1}"],
                volume=ticker_df[f"vol_{ticker1}"],
                fillna=True,
            )
            ticker_df[f"nvi_{ticker1}"] = nvi.negative_volume_index()
        except:
            ticker_df[f"nvi_{ticker1}"] = pd.Series(
                [0 for i in range(len(ticker_df[f"close_{ticker1}"]))]
            )

        obv = OnBalanceVolumeIndicator(
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            fillna=True,
        )

        vpt = VolumePriceTrendIndicator(
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            fillna=True,
        )

        wp = VolumeWeightedAveragePrice(
            high=ticker_df[f"high_{ticker1}"],
            low=ticker_df[f"low_{ticker1}"],
            close=ticker_df[f"close_{ticker1}"],
            volume=ticker_df[f"vol_{ticker1}"],
            window=10,
            fillna=True,
        )

        ticker_df[f"acc5_{ticker1}"] = acc.acc_dist_index()
        ticker_df[f"cmf5_{ticker1}"] = cmf.chaikin_money_flow()
        ticker_df[f"eom5_{ticker1}"] = eom.ease_of_movement()
        ticker_df[f"fii_{ticker1}"] = fii.force_index()

        ticker_df[f"obv_{ticker1}"] = obv.on_balance_volume()
        ticker_df[f"vpt_{ticker1}"] = vpt.volume_price_trend()
        ticker_df[f"wp_{ticker1}"] = wp.volume_weighted_average_price()

        from ta.momentum import (
            AwesomeOscillatorIndicator,
            ROCIndicator,
            RSIIndicator,
            StochRSIIndicator,
        )
        from ta.volatility import AverageTrueRange, BollingerBands
        from ta.trend import ADXIndicator

        aoi = AwesomeOscillatorIndicator(
            high=ticker_df[f"high_{ticker1}"],
            low=ticker_df[f"low_{ticker1}"],
            window1=5,
            window2=10,
            fillna=True,
        )

        roc = ROCIndicator(close=ticker_df[f"close_{ticker1}"], window=5, fillna=True)

        rsi = RSIIndicator(close=ticker_df[f"close_{ticker1}"], window=5, fillna=True)

        try:
            atr = AverageTrueRange(
                high=ticker_df[f"high_{ticker1}"],
                low=ticker_df[f"low_{ticker1}"],
                close=ticker_df[f"close_{ticker1}"],
                window=5,
                fillna=True,
            )
            ticker_df[f"atr_{ticker1}"] = atr.average_true_range()
        except:
            ticker_df[f"atr_{ticker1}"] = pd.Series(
                [0 for i in range(len(ticker_df[f"high_{ticker1}"]))]
            )

        bb = BollingerBands(close=ticker_df[f"close_{ticker1}"], window=10, fillna=True)

        try:
            adx5 = ADXIndicator(
                high=ticker_df[f"high_{ticker1}"],
                low=ticker_df[f"low_{ticker1}"],
                close=ticker_df[f"close_{ticker1}"],
                window=5,
                fillna=True,
            )

            adx10 = ADXIndicator(
                high=ticker_df[f"high_{ticker1}"],
                low=ticker_df[f"low_{ticker1}"],
                close=ticker_df[f"close_{ticker1}"],
                window=10,
                fillna=True,
            )

            ticker_df[f"adx5_{ticker1}"] = adx5.adx()
            ticker_df[f"adx10_{ticker1}"] = adx10.adx()
        except:
            ticker_df[f"adx5_{ticker1}"] = pd.Series(
                [0 for i in range(len(ticker_df[f"high_{ticker1}"]))]
            )
            ticker_df[f"adx10_{ticker1}"] = pd.Series(
                [0 for i in range(len(ticker_df[f"high_{ticker1}"]))]
            )

        ticker_df[f"aoi_{ticker1}"] = aoi.awesome_oscillator()
        ticker_df[f"roc_{ticker1}"] = roc.roc()
        ticker_df[f"rsi_{ticker1}"] = rsi.rsi()

        ticker_df[f"bhb_{ticker1}"] = bb.bollinger_hband()
        ticker_df[f"bhbi_{ticker1}"] = bb.bollinger_hband_indicator()
        ticker_df[f"blb_{ticker1}"] = bb.bollinger_lband()
        ticker_df[f"blbi_{ticker1}"] = bb.bollinger_lband_indicator()

        ticker_df[f"adx5_{ticker1}"] = adx5.adx()
        ticker_df[f"adx10_{ticker1}"] = adx10.adx()

        ticker_df = ticker_df.fillna(ticker_df.tail(100).median())
        ticker_dfs[ticker1] = ticker_df.iloc[-1]
        print(ticker_dfs[ticker1])
        # print(close)
        # break

    merged_df = pd.concat(ticker_dfs.values())

    merged_df = pd.DataFrame(merged_df).T

    target_names = [
        "vol_SI",
        "vol_RTS",
        "vol_IMOEX",
        "vol_GOLD",
        "volume_AFLT",
        "volume_CHMF",
        "volume_FEES",
        "volume_GAZP",
        "volume_GMKN",
        "volume_HYDR",
        "volume_IRAO",
        "volume_LKOH",
        "volume_MAGN",
        "volume_MTSS",
        "volume_NLMK",
        "volume_NVTK",
        "volume_ROSN",
        "volume_RTKM",
        "volume_SBERP",
        "volume_SNGSP",
        "volume_SNGS",
        "volume_TATN",
        "volume_VTBR",
        "volume_AFKS_M1",
        "volume_ALRS_M1",
        "volume_MOEX_M1",
        "volume_MTLR_M1",
    ]

    vol_columns = [
        "vol_SI",
        "vol_RTS",
        "vol_IMOEX",
        "vol_GOLD",
        "vol_AFLT",
        "vol_CHMF",
        "vol_FEES",
        "vol_GAZP",
        "vol_GMKN",
        "vol_HYDR",
        "vol_IRAO",
        "vol_LKOH",
        "vol_MAGN",
        "vol_MTSS",
        "vol_NLMK",
        "vol_NVTK",
        "vol_ROSN",
        "vol_RTKM",
        "vol_SBERP",
        "vol_SNGSP",
        "vol_SNGS",
        "vol_TATN",
        "vol_VTBR",
        "vol_AFKS_M1",
        "vol_ALRS_M1",
        "vol_MOEX_M1",
        "vol_MTLR_M1",
    ]

    # Построим отображение: vol_XXX -> volume_XXX (только если имя отличается)
    rename_dict = {
        old: new for old, new in zip(vol_columns, target_names) if old != new
    }
    merged_df.rename(columns=rename_dict, inplace=True)
    merged_df = merged_df.rename(
        columns={
            "close_minus_open_AFKS_M1": "close_minus_open_AFKS",
            "close_minus_open_ALRS_M1": "close_minus_open_ALRS",
            "close_minus_open_MOEX_M1": "close_minus_open_MOEX",
            "close_minus_open_MTLR_M1": "close_minus_open_MTLR",
        }
    )

    ohlcv.extend(
        [
            "sma_5_close_SI",
            "ema_5_close_SI",
            "sma_5_close_RTS",
            "ema_5_close_RTS",
            "sma_5_close_IMOEX",
            "ema_5_close_IMOEX",
            "sma_5_close_GOLD",
            "ema_5_close_GOLD",
            "sma_5_close_AFLT",
            "ema_5_close_AFLT",
            "sma_5_close_CHMF",  #  11
            "ema_5_close_CHMF",
            "sma_5_close_FEES",
            "ema_5_close_FEES",
            "sma_5_close_GAZP",
            "ema_5_close_GAZP",
            "sma_5_close_GMKN",
            "ema_5_close_GMKN",
            "sma_5_close_HYDR",
            "ema_5_close_HYDR",
            "sma_5_close_IRAO",
            "ema_5_close_IRAO",
            "sma_5_close_LKOH",
            "ema_5_close_LKOH",
            "sma_5_close_MAGN",
            "ema_5_close_MAGN",
            "sma_5_close_MTSS",
            "ema_5_close_MTSS",
            "sma_5_close_NLMK",
            "ema_5_close_NLMK",
            "sma_5_close_NVTK",
            "ema_5_close_NVTK",
            "sma_5_close_ROSN",
            "ema_5_close_ROSN",
            "sma_5_close_RTKM",
            "ema_5_close_RTKM",
            "sma_5_close_SBERP",
            "ema_5_close_SBERP",
            "sma_5_close_SNGSP",
            "ema_5_close_SNGSP",
            "sma_5_close_SNGS",
            "ema_5_close_SNGS",
            "sma_5_close_TATN",  # 32
            "ema_5_close_TATN",
            "sma_5_close_VTBR",
            "ema_5_close_VTBR",
            "sma_5_close_AFKS_M1",
            "ema_5_close_AFKS_M1",
            "sma_5_close_ALRS_M1",
            "ema_5_close_ALRS_M1",
            "sma_5_close_MOEX_M1",
            "ema_5_close_MOEX_M1",
            "sma_5_close_MTLR_M1",
            "ema_5_close_MTLR_M1",  # 11 = 54
            "close_minus_open_SI",
            "close_minus_open_RTS",
            "close_minus_open_IMOEX",
            "close_minus_open_GOLD",
            "close_minus_open_AFLT",
            "close_minus_open_CHMF",
            "close_minus_open_FEES",
            "close_minus_open_GAZP",
            "close_minus_open_GMKN",
            "close_minus_open_HYDR",
            "close_minus_open_IRAO",
            "close_minus_open_LKOH",
            "close_minus_open_MAGN",
            "close_minus_open_MTSS",
            "close_minus_open_NLMK",
            "close_minus_open_NVTK",
            "close_minus_open_ROSN",
            "close_minus_open_RTKM",
            "close_minus_open_SBERP",
            "close_minus_open_SNGSP",
            "close_minus_open_SNGS",
            "close_minus_open_TATN",
            "close_minus_open_VTBR",
            "close_minus_open_AFKS",
            "close_minus_open_ALRS",
            "close_minus_open_MOEX",
            "close_minus_open_MTLR",
        ]
    )  # 27

    osnova = ohlcv
    print("OSNOVA")
    print(osnova)
    joblib.dump(osnova, "osnova.joblib")
    print(merged_df.columns)
    joblib.dump(merged_df, "merged_df.joblib")

    df_osnova = merged_df[osnova]
    df_ost = merged_df.drop(osnova, axis=1)
    df_ost.drop(
        columns=["5min_std_AFKS_M1", "5min_std_ALRS_M1", "5min_std_MOEX_M1"],
        inplace=True,
        axis=1,
    )
    df_ost = df_ost.rename(columns={"5min_std_MTLR_M1": "5min_std_M1"})
    df_ost.drop(
        columns=["10min_std_AFKS_M1", "10min_std_ALRS_M1", "10min_std_MOEX_M1"],
        inplace=True,
        axis=1,
    )
    df_ost = df_ost.rename(columns={"10min_std_MTLR_M1": "10min_std_M1"})

    df_ost.drop(
        columns=[
            "down_streak_5min_AFKS_M1",
            "down_streak_5min_ALRS_M1",
            "down_streak_5min_MOEX_M1",
        ],
        inplace=True,
        axis=1,
    )
    df_ost = df_ost.rename(columns={"down_streak_5min_MTLR_M1": "down_streak_5min_M1"})
    df_ost.drop(
        columns=[
            "down_streak_10min_AFKS_M1",
            "down_streak_10min_ALRS_M1",
            "down_streak_10min_MOEX_M1",
        ],
        inplace=True,
        axis=1,
    )
    df_ost = df_ost.rename(
        columns={"down_streak_10min_MTLR_M1": "down_streak_10min_M1"}
    )

    df_ost.drop(
        columns=["mfi5_AFKS_M1", "mfi5_ALRS_M1", "mfi5_MOEX_M1"], inplace=True, axis=1
    )
    df_ost = df_ost.rename(columns={"mfi5_MTLR_M1": "mfi5_M1"})
    df_ost.drop(
        columns=["mfi10_AFKS_M1", "mfi10_ALRS_M1", "mfi10_MOEX_M1"],
        inplace=True,
        axis=1,
    )
    df_ost = df_ost.rename(columns={"mfi10_MTLR_M1": "mfi10_M1"})

    afks_cols = [i for i in df_ost.columns if "AFKS" in i]
    alrs_cols = [i for i in df_ost.columns if "ALRS" in i]
    moex_cols = [i for i in df_ost.columns if "MOEX" in i]

    df_ost.drop(columns=afks_cols, inplace=True, axis=1)
    df_ost.drop(columns=alrs_cols, inplace=True, axis=1)
    df_ost.drop(columns=moex_cols, inplace=True, axis=1)

    df_ost = df_ost.rename(
        columns={
            "nvi_MTLR_M1": "nvi_M1",
            "acc5_MTLR_M1": "acc5_M1",
            "cmf5_MTLR_M1": "cmf5_M1",
            "eom5_MTLR_M1": "eom5_M1",
            "fii_MTLR_M1": "fii_M1",
            "obv_MTLR_M1": "obv_M1",
            "vpt_MTLR_M1": "vpt_M1",
            "wp_MTLR_M1": "wp_M1",
            "atr_MTLR_M1": "atr_M1",
            "adx5_MTLR_M1": "adx5_M1",
            "adx10_MTLR_M1": "adx10_M1",
            "aoi_MTLR_M1": "aoi_M1",
            "roc_MTLR_M1": "roc_M1",
            "rsi_MTLR_M1": "rsi_M1",
            "bhb_MTLR_M1": "bhb_M1",
            "bhbi_MTLR_M1": "bhbi_M1",
            "blb_MTLR_M1": "blb_M1",
            "blbi_MTLR_M1": "blbi_M1",
        }
    )

    total_df = pd.concat([df_osnova, df_ost], axis=1)

    print(total_df.shape)

    from datetime import datetime

    now = datetime.now()

    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    total_df["Year"] = year
    total_df["Month"] = month
    total_df["Day"] = day
    total_df["Hour"] = hour
    total_df["Minute"] = minute
    joblib.dump(df_ost, "df_ost.joblib")

    imoex_df = merged_df[
        [
            "5min_std_IMOEX",
            "10min_std_IMOEX",
            "down_streak_5min_IMOEX",
            "down_streak_10min_IMOEX",
            "eom5_IMOEX",
            "fii_IMOEX",
            "aoi_IMOEX",
            "roc_IMOEX",
            "rsi_IMOEX",
            "atr_IMOEX",
            "bhb_IMOEX",
            "bhbi_IMOEX",
            "blb_IMOEX",
            "blbi_IMOEX",
            "adx5_IMOEX",
            "adx10_IMOEX",
        ]
    ]

    total_df = pd.concat([total_df, imoex_df], axis=1)
    print(total_df.shape)
    vmodel = total_df[cols]
    print(vmodel)

    return vmodel
    # from sklearn.ensemble import RandomForestRegressor
    # import sklearn
    # model = joblib.load('rf.joblib')

    # pred = model.predict(vmodel)

    # print(pred)

    # ema 'sma_5_close_SI', 'ema_5_close_SI', 'sma_5_close_RTS',
    #            'ema_5_close_RTS', 'sma_5_close_IMOEX', 'ema_5_close_IMOEX', 'sma_5_close_GOLD', 'ema_5_close_GOLD', 'sma_5_close_AFLT', 'ema_5_close_AFLT', 'sma_5_close_CHMF',
    #              'ema_5_close_CHMF', 'sma_5_close_FEES', 'ema_5_close_FEES', 'sma_5_close_GAZP', 'ema_5_close_GAZP', 'sma_5_close_GMKN', 'ema_5_close_GMKN', 'sma_5_close_HYDR',
    #                'ema_5_close_HYDR', 'sma_5_close_IRAO', 'ema_5_close_IRAO', 'sma_5_close_LKOH', 'ema_5_close_LKOH', 'sma_5_close_MAGN', 'ema_5_close_MAGN', 'sma_5_close_MTSS',
    #                'ema_5_close_MTSS', 'sma_5_close_NLMK', 'ema_5_close_NLMK', 'sma_5_close_NVTK', 'ema_5_close_NVTK', 'sma_5_close_ROSN', 'ema_5_close_ROSN', 'sma_5_close_RTKM',
    #                'ema_5_close_RTKM', 'sma_5_close_SBERP', 'ema_5_close_SBERP', 'sma_5_close_SNGSP', 'ema_5_close_SNGSP', 'sma_5_close_SNGS', 'ema_5_close_SNGS', 'sma_5_close_TATN',
    #                  'ema_5_close_TATN', 'sma_5_close_VTBR', 'ema_5_close_VTBR', 'sma_5_close_AFKS_M1', 'ema_5_close_AFKS_M1', 'sma_5_close_ALRS_M1', 'ema_5_close_ALRS_M1',
    #                    'sma_5_close_MOEX_M1', 'ema_5_close_MOEX_M1', 'sma_5_close_MTLR_M1', 'ema_5_close_MTLR_M1',

    # 'close_minus_open_SI', 'close_minus_open_RTS', 'close_minus_open_IMOEX',
    #                      'close_minus_open_GOLD', 'close_minus_open_AFLT', 'close_minus_open_CHMF', 'close_minus_open_FEES', 'close_minus_open_GAZP', 'close_minus_open_GMKN',
    #                        'close_minus_open_HYDR', 'close_minus_open_IRAO', 'close_minus_open_LKOH', 'close_minus_open_MAGN', 'close_minus_open_MTSS', 'close_minus_open_NLMK',
    #                        'close_minus_open_NVTK', 'close_minus_open_ROSN', 'close_minus_open_RTKM', 'close_minus_open_SBERP', 'close_minus_open_SNGSP', 'close_minus_open_SNGS',
    #                        'close_minus_open_TATN', 'close_minus_open_VTBR', 'close_minus_open_AFKS', 'close_minus_open_ALRS', 'close_minus_open_MOEX', 'close_minus_open_MTLR',

    #                          '5min_std_SI', '10min_std_SI', '5min_std_RTS', '10min_std_RTS', '5min_std_IMOEX', '10min_std_IMOEX', '5min_std_GOLD', '10min_std_GOLD', '5min_std_AFLT',
    #                            '10min_std_AFLT', '5min_std_CHMF', '10min_std_CHMF', '5min_std_FEES', '10min_std_FEES', '5min_std_GAZP', '10min_std_GAZP', '5min_std_GMKN',
    #                            '10min_std_GMKN', '5min_std_HYDR', '10min_std_HYDR', '5min_std_IRAO', '10min_std_IRAO', '5min_std_LKOH', '10min_std_LKOH', '5min_std_MAGN',
    #                            '10min_std_MAGN', '5min_std_MTSS', '10min_std_MTSS', '5min_std_NLMK', '10min_std_NLMK', '5min_std_NVTK', '10min_std_NVTK', '5min_std_ROSN',
    #                            '10min_std_ROSN', '5min_std_RTKM', '10min_std_RTKM', '5min_std_SBERP', '10min_std_SBERP', '5min_std_SNGSP', '10min_std_SNGSP', '5min_std_SNGS',
    #                            '10min_std_SNGS', '5min_std_TATN', '10min_std_TATN', '5min_std_VTBR', '10min_std_VTBR', '5min_std_M1', '10min_std_M1',
    #
    #                           'down_streak_5min_SI',
    #                            'down_streak_10min_SI', 'down_streak_5min_RTS', 'down_streak_10min_RTS', 'down_streak_5min_IMOEX', 'down_streak_10min_IMOEX', 'down_streak_5min_GOLD',
    #                              'down_streak_10min_GOLD', 'down_streak_5min_AFLT', 'down_streak_10min_AFLT', 'down_streak_5min_CHMF', 'down_streak_10min_CHMF', 'down_streak_5min_FEES',
    #                                'down_streak_10min_FEES', 'down_streak_5min_GAZP', 'down_streak_10min_GAZP', 'down_streak_5min_GMKN', 'down_streak_10min_GMKN', 'down_streak_5min_HYDR',
    #                                  'down_streak_10min_HYDR', 'down_streak_5min_IRAO', 'down_streak_10min_IRAO', 'down_streak_5min_LKOH', 'down_streak_10min_LKOH', 'down_streak_5min_MAGN',
    #                                    'down_streak_10min_MAGN', 'down_streak_5min_MTSS', 'down_streak_10min_MTSS', 'down_streak_5min_NLMK', 'down_streak_10min_NLMK', 'down_streak_5min_NVTK',
    #                                      'down_streak_10min_NVTK', 'down_streak_5min_ROSN', 'down_streak_10min_ROSN', 'down_streak_5min_RTKM', 'down_streak_10min_RTKM', 'down_streak_5min_SBERP',
    #                                        'down_streak_10min_SBERP', 'down_streak_5min_SNGSP', 'down_streak_10min_SNGSP', 'down_streak_5min_SNGS', 'down_streak_10min_SNGS',
    #                                        'down_streak_5min_TATN', 'down_streak_10min_TATN', 'down_streak_5min_VTBR', 'down_streak_10min_VTBR', 'down_streak_5min_M1', 'down_streak_10min_M1',

    #                                          'mfi5_SI', 'mfi10_SI', 'mfi5_RTS', 'mfi10_RTS', 'mfi5_GOLD', 'mfi10_GOLD', 'mfi5_AFLT', 'mfi10_AFLT', 'mfi5_CHMF', 'mfi10_CHMF', 'mfi5_FEES',
    #                                          'mfi10_FEES', 'mfi5_GAZP', 'mfi10_GAZP', 'mfi5_GMKN', 'mfi10_GMKN', 'mfi5_HYDR', 'mfi10_HYDR', 'mfi10_IRAO', 'mfi5_LKOH', 'mfi10_LKOH', 'mfi5_MAGN',
    #                                          'mfi10_MAGN', 'mfi10_MTSS', 'mfi5_NLMK', 'mfi10_NLMK', 'mfi5_NVTK', 'mfi10_NVTK', 'mfi5_ROSN', 'mfi10_ROSN', 'mfi5_RTKM', 'mfi10_RTKM', 'mfi5_SBERP',
    #                                            'mfi10_SBERP', 'mfi5_SNGSP', 'mfi10_SNGSP', 'mfi5_SNGS', 'mfi10_SNGS', 'mfi10_TATN', 'mfi5_VTBR', 'mfi10_VTBR', 'mfi5_M1', 'mfi10_M1',

    #                                              'eom5_SI', 'fii_SI', 'obv_SI', 'wp_SI', 'eom5_RTS', 'fii_RTS', 'obv_RTS', 'wp_RTS', 'eom5_IMOEX', 'fii_IMOEX', 'fii_GOLD', 'obv_GOLD',
    #                                                'wp_GOLD', 'fii_AFLT', 'obv_AFLT', 'wp_AFLT', 'eom5_CHMF', 'fii_CHMF', 'obv_CHMF', 'wp_CHMF', 'fii_FEES', 'obv_FEES', 'wp_FEES', 'eom5_GAZP', 'fii_GAZP',
    # 'obv_GAZP', 'wp_GAZP', 'fii_GMKN', 'obv_GMKN', 'wp_GMKN', 'fii_HYDR', 'obv_HYDR', 'wp_HYDR', 'fii_IRAO', 'obv_IRAO', 'wp_IRAO', 'eom5_LKOH', 'fii_LKOH', 'obv_LKOH',
    # 'wp_LKOH', 'eom5_MAGN', 'fii_MAGN', 'obv_MAGN', 'wp_MAGN', 'fii_MTSS', 'obv_MTSS', 'wp_MTSS', 'eom5_NLMK', 'fii_NLMK', 'obv_NLMK', 'wp_NLMK', 'eom5_NVTK', 'fii_NVTK', 'obv_NVTK',
    # 'wp_NVTK', 'eom5_ROSN', 'fii_ROSN', 'obv_ROSN', 'wp_ROSN', 'fii_RTKM', 'obv_RTKM', 'wp_RTKM', 'eom5_SBERP', 'fii_SBERP', 'obv_SBERP', 'wp_SBERP', 'fii_SNGSP', 'obv_SNGSP', 'wp_SNGSP',
    # 'fii_SNGS', 'obv_SNGS', 'wp_SNGS', 'eom5_TATN', 'fii_TATN', 'obv_TATN', 'wp_TATN', 'eom5_VTBR', 'fii_VTBR', 'obv_VTBR', 'wp_VTBR', 'fii_M1', 'obv_M1', 'wp_M1', 'aoi_SI', 'rsi_SI',
    #   'atr_SI', 'bhb_SI', 'bhbi_SI', 'blb_SI', 'blbi_SI', 'adx5_SI', 'adx10_SI', 'aoi_RTS', 'rsi_RTS', 'atr_RTS', 'bhb_RTS', 'bhbi_RTS', 'blb_RTS', 'blbi_RTS', 'adx5_RTS', 'adx10_RTS',
    #     'aoi_IMOEX', 'roc_IMOEX', 'rsi_IMOEX', 'atr_IMOEX', 'bhb_IMOEX', 'bhbi_IMOEX', 'blb_IMOEX', 'blbi_IMOEX', 'adx5_IMOEX', 'adx10_IMOEX', 'aoi_GOLD', 'rsi_GOLD', 'atr_GOLD', 'bhb_GOLD',
    #     'bhbi_GOLD', 'blb_GOLD', 'blbi_GOLD', 'adx5_GOLD', 'adx10_GOLD', 'aoi_AFLT', 'rsi_AFLT', 'atr_AFLT', 'bhb_AFLT', 'bhbi_AFLT', 'blb_AFLT', 'blbi_AFLT', 'adx5_AFLT', 'adx10_AFLT',
    #     'aoi_CHMF', 'rsi_CHMF', 'atr_CHMF', 'bhb_CHMF', 'bhbi_CHMF', 'blb_CHMF', 'blbi_CHMF', 'adx5_CHMF', 'adx10_CHMF', 'aoi_FEES', 'rsi_FEES', 'atr_FEES', 'bhb_FEES', 'bhbi_FEES', 'blb_FEES',
    #     'blbi_FEES', 'adx5_FEES', 'adx10_FEES', 'aoi_GAZP', 'rsi_GAZP', 'atr_GAZP', 'bhb_GAZP', 'bhbi_GAZP', 'blb_GAZP', 'blbi_GAZP', 'adx5_GAZP', 'adx10_GAZP', 'aoi_GMKN', 'rsi_GMKN',
    #     'atr_GMKN', 'bhb_GMKN', 'bhbi_GMKN', 'blb_GMKN', 'blbi_GMKN', 'adx5_GMKN', 'adx10_GMKN', 'aoi_HYDR', 'rsi_HYDR', 'atr_HYDR', 'bhb_HYDR', 'bhbi_HYDR', 'blb_HYDR', 'blbi_HYDR',
    #     'adx5_HYDR', 'adx10_HYDR', 'aoi_IRAO', 'rsi_IRAO', 'atr_IRAO', 'bhb_IRAO', 'bhbi_IRAO', 'blb_IRAO', 'blbi_IRAO', 'adx5_IRAO', 'adx10_IRAO', 'aoi_LKOH', 'rsi_LKOH', 'atr_LKOH',
    #       'bhb_LKOH', 'bhbi_LKOH', 'blb_LKOH', 'blbi_LKOH', 'adx5_LKOH', 'adx10_LKOH', 'aoi_MAGN', 'rsi_MAGN', 'atr_MAGN', 'bhb_MAGN', 'bhbi_MAGN', 'blb_MAGN', 'blbi_MAGN', 'adx5_MAGN',
    #         'adx10_MAGN', 'aoi_MTSS', 'rsi_MTSS', 'atr_MTSS', 'bhb_MTSS', 'bhbi_MTSS', 'blb_MTSS', 'blbi_MTSS', 'adx5_MTSS', 'adx10_MTSS', 'aoi_NLMK', 'rsi_NLMK', 'atr_NLMK', 'bhb_NLMK',
    #         'bhbi_NLMK', 'blb_NLMK', 'blbi_NLMK', 'adx5_NLMK', 'adx10_NLMK', 'aoi_NVTK', 'rsi_NVTK', 'atr_NVTK', 'bhb_NVTK', 'bhbi_NVTK', 'blb_NVTK', 'blbi_NVTK', 'adx5_NVTK', 'adx10_NVTK',
    #           'aoi_ROSN', 'rsi_ROSN', 'atr_ROSN', 'bhb_ROSN', 'bhbi_ROSN', 'blb_ROSN', 'blbi_ROSN', 'adx5_ROSN', 'adx10_ROSN', 'aoi_RTKM', 'rsi_RTKM', 'atr_RTKM',
    # 'bhb_RTKM', 'bhbi_RTKM', 'blb_RTKM', 'blbi_RTKM', 'adx5_RTKM', 'adx10_RTKM', 'aoi_SBERP', 'rsi_SBERP', 'atr_SBERP', 'bhb_SBERP', 'bhbi_SBERP', 'blb_SBERP', 'blbi_SBERP',
    #   'adx5_SBERP', 'adx10_SBERP', 'aoi_SNGSP', 'rsi_SNGSP', 'atr_SNGSP', 'bhb_SNGSP', 'bhbi_SNGSP', 'blb_SNGSP', 'blbi_SNGSP', 'adx5_SNGSP', 'adx10_SNGSP', 'aoi_SNGS', 'rsi_SNGS',
    #   'atr_SNGS', 'bhb_SNGS', 'bhbi_SNGS', 'blb_SNGS', 'blbi_SNGS', 'adx5_SNGS', 'adx10_SNGS', 'aoi_TATN', 'rsi_TATN', 'atr_TATN', 'bhb_TATN', 'bhbi_TATN',
    # 'blb_TATN', 'blbi_TATN', 'adx5_TATN', 'adx10_TATN', 'aoi_VTBR', 'rsi_VTBR', 'atr_VTBR', 'bhb_VTBR', 'bhbi_VTBR', 'blb_VTBR', 'blbi_VTBR', 'adx5_VTBR', 'adx10_VTBR',
    # 'aoi_M1', 'rsi_M1', 'atr_M1', 'bhb_M1', 'bhbi_M1', 'blb_M1', 'blbi_M1', 'adx5_M1', 'adx10_M1']


if __name__ == "__main__":
    df = get_current_features1()
    print(df)
