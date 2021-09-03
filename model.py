import datetime
from collections import Counter
from random import seed

from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_taiwan_stock_price_data():
    df = pd.read_csv("taiwan_stock_price.csv")
    df["stock_id"] = df["stock_id"].astype(str)
    return df


def clean_df(df):
    df = df[df["Trading_Volume"] > 0]
    df = df.sort_values("date")
    df["stock_id"] = df["stock_id"].astype(str)
    logger.info(df["Trading_Volume"].describe()["mean"])
    df = df[df["Trading_Volume"] > df["Trading_Volume"].describe()["mean"]]
    df = df.reset_index(drop=True)
    return df


def create_target(spread_per):
    if spread_per > 0.03:
        return 1
    elif spread_per < -0.03:
        return -1
    else:
        return 0


def feature_engineer(df: pd.DataFrame, last_days: int = 7):
    # create target
    logger.info("feature_engineer")
    df["spread_per"] = df["spread"] / df["open"]
    df["target"] = df["spread_per"].apply(create_target)
    logger.info(f'target: {Counter(df["target"])}')

    # create feature
    for day in range(1, last_days + 1):
        df[f"open_minus_{day}"] = df.groupby("stock_id")["open"].shift(day)
        df[f"max_minus_{day}"] = df.groupby("stock_id")["max"].shift(day)
        df[f"min_minus_{day}"] = df.groupby("stock_id")["min"].shift(day)
        df[f"close_minus_{day}"] = df.groupby("stock_id")["close"].shift(day)
        df[f"Trading_Volume_minus_{day}"] = df.groupby("stock_id")[
            "Trading_Volume"
        ].shift(day)
        df[f"Trading_turnover_minus_{day}"] = df.groupby("stock_id")[
            "Trading_turnover"
        ].shift(day)
        df[f"spread_per_{day}"] = df.groupby("stock_id")["spread_per"].shift(
            day
        )

    df = df.dropna()
    return df


def split_train_test(df):
    logger.info("split_train_test")
    train_test_split_index = int(len(df) * 0.7)
    train_df = df[:train_test_split_index]
    test_df = df[train_test_split_index:]
    return train_df, test_df


def create_feature_variable(df, last_days: int = 7):
    logger.info("create_feature_variable")
    colname_list = [
        [
            f"open_minus_{day}",
            f"max_minus_{day}",
            f"min_minus_{day}",
            f"close_minus_{day}",
            f"Trading_Volume_minus_{day}",
            f"Trading_turnover_minus_{day}",
            # f"spread_per_{day}",
        ]
        for day in range(1, last_days + 1)
    ]
    feature_list = []
    [feature_list.extend(col) for col in colname_list]
    feature_list = [
        feature for feature in feature_list if feature in list(df.columns)
    ]
    return feature_list


def create_model(train_df, test_df, feature_list):
    logger.info("create_model")
    logger.info(f"feature: {feature_list}")
    x_train = train_df[feature_list]
    x_test = test_df[feature_list]
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    watchlist = [(x_train, y_train), (x_test, y_test)]
    seed(100)
    clf = xgb.XGBClassifier(
        eta=0.1,
        max_depth=5,
        eval_metric="merror",
        n_estimators=100,
        verbose=True,
        num_classes=3,
        # min_child_weight=0.5,
        # subsample =0.5,
        # alpha=1,
    )
    clf.fit(x_train, y_train, eval_set=watchlist, early_stopping_rounds=5)

    train_pred = clf.predict(x_train)
    train_df["pred"] = train_pred
    train_array = confusion_matrix(train_df["target"], train_df["pred"])
    train_confusion_matrix = pd.DataFrame(
        train_array, range(-1, 2), range(-1, 2)
    )
    logger.info(f"train_confusion_matrix \n {train_confusion_matrix}")

    test_pred = clf.predict(x_test)
    test_df["pred"] = test_pred
    test_array = confusion_matrix(test_df["target"], test_df["pred"])
    test_confusion_matrix = pd.DataFrame(test_array, range(-1, 2), range(-1, 2))
    logger.info(f"test_confusion_matrix \n {test_confusion_matrix}")
    return train_df, test_df


def calculate_return_profit(pred_df):
    pred_1 = pred_df[pred_df["pred"] == 1]
    pred_1["profit"] = pred_1["close"] - pred_1["open"]
    pred_1 = pred_1[["open", "date", "profit"]]

    pred_minut_1 = pred_df[pred_df["pred"] == -1]
    pred_minut_1["profit"] = pred_minut_1["open"] - pred_minut_1["close"]
    pred_minut_1 = pred_minut_1[["open", "date", "profit"]]

    pred_df = pd.concat([pred_1, pred_minut_1], axis=0)

    mean_df = pred_df.groupby("date")["profit"].mean()
    mean_df = mean_df.reset_index()

    open_df = pred_df.groupby("date")["open"].sum()
    open_df = open_df.reset_index()

    count_df = pred_df.groupby("date")["open"].count()
    count_df = count_df.reset_index()
    count_df.columns = ["date", "deal_times"]

    pred_df = mean_df.merge(open_df, on=["date"]).merge(count_df, on=["date"])
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df["year"] = pred_df["date"].dt.year
    pred_df["fee"] = pred_df["open"] * 0.001425 * 2 * 0.5
    pred_df["tax"] = pred_df["open"] * 0.003 * 0.5
    pred_df["profit_fee_tax"] = (
        pred_df["profit"] - pred_df["fee"] - pred_df["tax"]
    )
    pred_df["total_profit_fee_tax"] = (
        pred_df["profit_fee_tax"].cumsum()
    ).round(2)
    pred_df["total_profit"] = (pred_df["profit_fee_tax"].cumsum()).round(2)
    for col in [
        "fee",
        "tax",
        "profit_fee_tax",
        "total_profit_fee_tax",
        "total_profit",
    ]:
        pred_df[col] = pred_df[col].round(2)
    return pred_df


def create_win_rate(profit_df, profit_col):
    win_rate = (
        round(
            len(profit_df[profit_df[profit_col] > 0]) / len(profit_df),
            2,
        )
        * 100
    )
    return win_rate


def show_profit(profit_df, text: str):
    train_win_rate = round(create_win_rate(profit_df, "profit"), 2)
    train_win_rate_with_fee_tax = round(
        create_win_rate(profit_df, "profit_fee_tax"), 2
    )
    profit_df["return"] = [
        0 if x > 0 else x for x in list(profit_df["profit_fee_tax"])
    ]
    max_return = 0
    max_ret = 0
    for ret in list(profit_df["return"]):
        if ret == 0:
            max_ret = 0
        max_ret += ret
        if max_ret < max_return:
            max_return = max_ret
    logger.info(
        f"""
    {text}
    ---------------------------------------------------------------------------
    平均資金: {round(profit_df["open"].mean(),2)*1000}
    平均每天交易資金: {round(profit_df['open'].mean() ,2)*1000}
    最大交易資金: {round(profit_df['open'].max() ,2)*1000}
    平均每天交易次數: {round(profit_df['deal_times'].mean() ,2)}
    最大每天交易次數: {round(profit_df['deal_times'].max() ,2)}
    最小每天交易次數: {round(profit_df['deal_times'].min() ,2)}
    ---------------------------------------------------------------------------
    平均報酬: {round(profit_df["profit"].mean(),2)*1000}
    平均年報酬: {profit_df.groupby('year')['profit'].sum().mean().round(2)*1000}
    勝率: {train_win_rate}%
    ---------------------------------------------------------------------------
    含手續費 0.001425、交易稅 0.003*0.5
    平均報酬: {round(profit_df["profit_fee_tax"].mean(),2)*1000}
    最大單次損失: {round(profit_df["profit_fee_tax"].min(),2)*1000}
    最大累積損失: {round(max_return,2)*1000}
    平均年報酬: {profit_df.groupby('year')['profit_fee_tax'].sum().mean().round(2)*1000}
    勝率: {train_win_rate_with_fee_tax}%
    """
    )


def main():
    stock_price_df = load_taiwan_stock_price_data()
    stock_price_df = clean_df(stock_price_df)
    last_days = 5
    df = feature_engineer(stock_price_df, last_days=last_days)
    train_df, test_df = split_train_test(df)
    feature_list = create_feature_variable(df, last_days=last_days)
    train_df, test_df = create_model(train_df, test_df, feature_list)
    train_profit_df = calculate_return_profit(train_df)
    test_profit_df = calculate_return_profit(test_df)

    show_profit(train_profit_df, "train")
    show_profit(test_profit_df, "test")

    test_profit_df.to_csv('test_profit_df.csv', index=False)
