import pandas as pd
import numpy as np
import yfinance as yf
import talib

from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#constants and utilities
TRADING_DAYS = 21
RFR = 0.05
TARGET_SRC = "growth_63_d"
TARGET = "is_growth_63"
#technical indicators name references
#list of momentum indicators
momentum_indicators = [
    "adx", "adxr", "apo", "aroonosc", "bop", "cci", "cmo", "dx",
    "macd", "mfi", "minus_di", "mom", "plus_di", "plus_dm", "ppo",
    "roc", "rocp", "rocr", "rocr100", "rsi", "trix", "ultosc", "willr"
]
#list of volume indicators
volume_indicators = ["ad", "adosc"]
#list of volatility indicators
volatility_indicators = ["atr", "natr"]
#list of cycle indicators
cycle_indicators = [
    "ht_dcperiod", "ht_dcphase", "ht_sine", "ht_leadsine", "ht_trendmod"
]
#list of price transforms indicators
price_indicators = ["avgprice", "medprice", "wclprice"]

#download data from yf
def get_ticker_data(tickers: list[str]=["FMTY14.MX"]) -> pd.DataFrame:
    data = pd.DataFrame()
    for ticker in tickers:
        #get ticker available daily data
        ticker_yf = yf.Ticker(ticker)
        ticker_data = ticker_yf.history(period="max", interval="1d")
        #format date
        ticker_data["date"] = pd.to_datetime(ticker_data.index, format="%Y-%m-%d")
        #assign to data
        if data.empty:
            data = ticker_data
        else:
            data = pd.concat([data, ticker_data], ignore_index=True)
    return data

#apply cleaning ops to df
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    #turn index into a feature
    df = df.reset_index(drop=True)
    #change dtype of features
    df["Volume"] = df["Volume"].astype("float64")
    #drop undesired features
    df = df.drop(columns=["Stock Splits"])
    #remove duplicates if any
    #print(df.shape)
    #df = df.drop_duplicates()
    #print(df.shape)
    #remove observations if all features are NaN
    df = df.dropna(how="all")
    return df

def get_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    #define future growth periods
    growth_periods = [TRADING_DAYS * month for month in range(1, 13)]
    for days in growth_periods:
        period = "growth_"+str(days)+"_d"
        df[period] = df["Close"].shift(-days) / df["Close"]
    #features required to get indicators
    openn = df["Open"].values
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    vol = df["Volume"].values
    #momentum
    df["adx_21"] = talib.ADX(high, low, close, timeperiod=TRADING_DAYS)
    df["adxr_21"] = talib.ADXR(high, low, close, timeperiod=TRADING_DAYS)
    df["apo_21_63"] = talib.APO(
        close, fastperiod=TRADING_DAYS, slowperiod=TRADING_DAYS * 3, matype=1
    )
    df["aroonosc_21"] = talib.AROONOSC(high, low, timeperiod=TRADING_DAYS)
    df["aroonosc_63"] = talib.AROONOSC(high, low, timeperiod=TRADING_DAYS * 3)
    df["bop"] = talib.BOP(openn, high, low, close)
    df["cci_21"] = talib.CCI(high, low, close, timeperiod=TRADING_DAYS)
    df["cmo_21"] = talib.CMO(close, timeperiod=TRADING_DAYS)
    df["dx_21"] = talib.DX(high, low, close, timeperiod=TRADING_DAYS)
    df["cci_63"] = talib.CCI(high, low, close, timeperiod=TRADING_DAYS * 3)
    df["cmo_63"] = talib.CMO(close, timeperiod=TRADING_DAYS * 3)
    df["dx_63"] = talib.DX(high, low, close, timeperiod=TRADING_DAYS * 3)
    df["macd_21"], df["macd_sig_21"], df["macd_hist_21"] = talib.MACD(
        close, fastperiod=TRADING_DAYS, slowperiod=TRADING_DAYS * 3, signalperiod=TRADING_DAYS
    )
    df["macd_42"], df["macd_sig_42"], df["macd_hist_42"] = talib.MACD(
        close, fastperiod=TRADING_DAYS * 2, slowperiod=TRADING_DAYS * 6, signalperiod=TRADING_DAYS
    )
    df["mfi_21"] = talib.MFI(high, low, close, vol, timeperiod=TRADING_DAYS)
    df["minus_di_21"] = talib.MINUS_DI(high, low, close, timeperiod=TRADING_DAYS)
    df["mom_21"] = talib.MOM(close, timeperiod=TRADING_DAYS)
    df["plus_di_21"] = talib.PLUS_DI(high, low, close, timeperiod=TRADING_DAYS)
    df["plus_dm_21"] = talib.PLUS_DM(high, low, timeperiod=TRADING_DAYS)
    df["mfi_63"] = talib.MFI(high, low, close, vol, timeperiod=TRADING_DAYS * 3)
    df["minus_di_63"] = talib.MINUS_DI(high, low, close, timeperiod=TRADING_DAYS * 3)
    df["mom_63"] = talib.MOM(close, timeperiod=TRADING_DAYS * 3)
    df["plus_di_63"] = talib.PLUS_DI(high, low, close, timeperiod=TRADING_DAYS * 3)
    df["plus_dm_63"] = talib.PLUS_DM(high, low, timeperiod=TRADING_DAYS * 3)
    df["ppo_21_63"] = talib.PPO(close, fastperiod=TRADING_DAYS, slowperiod=TRADING_DAYS * 3, matype=1)
    df["ppo_42_126"] = talib.PPO(close, fastperiod=TRADING_DAYS * 2, slowperiod=TRADING_DAYS * 6, matype=1)
    df["roc_21"] = talib.ROC(close, timeperiod=TRADING_DAYS)
    df["rocp_21"] = talib.ROCP(close, timeperiod=TRADING_DAYS)
    df["rocr_21"] = talib.ROCR(close, timeperiod=TRADING_DAYS)
    df["rocr100_21"] = talib.ROCR100(close, timeperiod=TRADING_DAYS)
    df["rsi_21"] = talib.RSI(close, timeperiod=TRADING_DAYS)
    df["trix_21"] = talib.TRIX(close, timeperiod=TRADING_DAYS)
    df["roc_63"] = talib.ROC(close, timeperiod=TRADING_DAYS * 3)
    df["rocp_63"] = talib.ROCP(close, timeperiod=TRADING_DAYS * 3)
    df["rocr_63"] = talib.ROCR(close, timeperiod=TRADING_DAYS * 3)
    df["rocr100_63"] = talib.ROCR100(close, timeperiod=TRADING_DAYS * 3)
    df["rsi_63"] = talib.RSI(close, timeperiod=TRADING_DAYS * 3)
    df["trix_63"] = talib.TRIX(close, timeperiod=TRADING_DAYS * 3)
    df["ultosc_21"] = talib.ULTOSC(
        high, low, close, timeperiod1=TRADING_DAYS, timeperiod2=TRADING_DAYS * 2, timeperiod3=TRADING_DAYS * 4
    )
    df["ultosc_63"] = talib.ULTOSC(
        high, low, close, timeperiod1=TRADING_DAYS, timeperiod2=TRADING_DAYS * 3, timeperiod3=TRADING_DAYS * 6
    )
    df["willr_21"] = talib.WILLR(high, low, close, timeperiod=TRADING_DAYS)
    df["willr_63"] = talib.WILLR(high, low, close, timeperiod=TRADING_DAYS * 3)
    #volume
    df["ad"] = talib.AD(high, low, close, vol)
    df["adosc_21"] = talib.ADOSC(
        high, low, close, vol, fastperiod=TRADING_DAYS, slowperiod=TRADING_DAYS * 3
    )
    df["adosc_63"] = talib.ADOSC(
        high, low, close, vol, fastperiod=TRADING_DAYS * 2, slowperiod=TRADING_DAYS * 6
    )
    #volatility
    df["atr_21"] = talib.ATR(high, low, close, timeperiod=TRADING_DAYS)
    df["natr_21"] = talib.NATR(high, low, close, timeperiod=TRADING_DAYS)
    df["natr_63"] = talib.NATR(high, low, close, timeperiod=TRADING_DAYS * 3)
    #cycle
    df["ht_dcperiod"] = talib.HT_DCPERIOD(close)
    df["ht_dcphase"] = talib.HT_DCPHASE(close)
    df["ht_sine"], df["ht_leadsine"] = talib.HT_SINE(close)
    df["ht_trendmod"]  = talib.HT_TRENDMODE(close)
    #price
    df["avgprice"] = talib.AVGPRICE(openn, high, low, close)
    df["medprice"] = talib.MEDPRICE(high, low)
    df["wclprice"] = talib.WCLPRICE(high, low, close)
    #other
    df["volatility"] = df["Close"].rolling(30).std() * np.sqrt(252)
    df["sharpe"] = (df["growth_252_d"] - RFR) / df["volatility"]
    #to predict feature
    df[TARGET] = (df.growth_63_d > 1).astype(int)
    #create week of month from date feature
    df["wom"] = df.date.dt.day.apply(
        lambda day : str((day - 1) // 7 + 1)
    )
    #use month name from date
    df["month"] = df["date"].dt.strftime("%b").str.lower()
    #define new feature month_wom
    df["month_wom"] = df["month"] + "_w" + df["wom"]
    #define day of week from date
    df["dow"] = df["date"].dt.strftime("%a").str.lower()
    #remove aux column wom
    df = df.drop(columns=["wom"])
    print("wom in fmty_df", "wom" in df.columns)
    #change dtype of ht_trendmod
    df["ht_trendmod"] = df["ht_trendmod"].astype("float64")
    #return modified df
    return df

#return dummy data from df selected features
def get_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    #categorical features
    cat = ["month", "dow", "month_wom"]
    #get dummies from cat
    dummy_data = pd.get_dummies(df[cat], dtype="int32")
    #return dummies df
    return dummy_data

def get_corr_analysis(df: pd.DataFrame) -> pd.DataFrame:
    num_features = []
    cat_features = []
    other_features = []
    #identify features for analysis
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    #allow search using str to include/exclude easily
    with pd.option_context("future.infer_string", True):
        cat_features = df.select_dtypes(include=["str"]).columns.tolist()
    with pd.option_context("future.infer_string", True):
        other_features = df.select_dtypes(exclude=[np.number, "str"]).columns.tolist()
    #print identified features
    print("features in df: ", df.shape[1])
    print("numerical_features: ", len(num_features))
    print("categorical_features: ", len(cat_features))
    print("other_features: ", len(other_features))

    #correlation with TARGET, sort desc
    corr_res = df[num_features].corr().sort_values(TARGET, ascending=False)
    #select target variable to apply month_wom rule_filter
    corr_res_df = corr_res[TARGET].copy()
    #use index as column
    corr_res_df_tg = corr_res_df.reset_index()
    #rename columns
    corr_res_df_tg.columns = ["dummy_feature", "corr_val"]
    #rule_filter month_wom features
    month_wom_rule_filter = corr_res_df_tg[
        corr_res_df_tg["dummy_feature"].str.contains("month_wom")
    ]
    #round to 3 decimal places
    month_wom_rule_filter["corr_val"] = month_wom_rule_filter["corr_val"].round(3)
    #return correlation result
    return month_wom_rule_filter

#split df observations in train, val, test, similar to code from module 3
def get_splits(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp, 
    train_sz=0.7, val_sz=0.15) -> list[str]:
    #define start and end ensuring use of time units
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_sz)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_sz)
    #define splits
    split_labels = []
    for date in df["date"]:
        if date <= train_end:
            split_labels.append("train")
        elif date <= val_end:
            split_labels.append("val")
        else:
            split_labels.append("test")
    #return assigned split labels to be assigned to a df feature
    return split_labels

#define hand rules
def get_hand_rules(df: pd.DataFrame) -> pd.DataFrame:
    #indicators to choose from to define hand rules
    selected_indicators = [
        "cci_21", "cci_63", "cmo_21", "cmo_63",
        "adosc_21", "adosc_63", "medprice", "wclprice"
    ]
    print("Display descriptive statistics from indicators for user reference")
    print(df[selected_indicators].describe())
    #create hand rules
    #−100 to +100 price is in its "normal" range
    df["rl_0_cci_63"] = ((df.cci_63 > -110) & (df.cci_63 < 105)).astype(int)
    #−50 to +50 price is in its "normal" range
    df["rl_1_cmo_63"] = ((df.cmo_63 > -50) & (df.cmo_63 < 50)).astype(int)
    #adosc > 0 buy accelerates, adosc < 0 sell accelerates, ~0 balanced
    df["rl_2_cci_adosc_63"] = (
        (df.cci_63 > -120) & (df.cci_63 < 100) & (df.adosc_63 > 0)
    ).astype(int)
    df["rl_3_cmo_adosc_63"] = (
        (df.cci_63 > -50) & (df.cci_63 < 50) & (df.adosc_63 > 0)
    ).astype(int)
    #return df with added hand rules
    return df

#review hand rules precision, similar to code from module 3
def show_hand_rules_precision(df: pd.DataFrame):
    #list of hand rules
    rules = [pred_name for pred_name in df.keys() if pred_name.startswith("rl")]
    #create column is_correct from each rule
    for rule in rules:
        prefix = rule.split("_")[1]
        df[f"is_correct_{prefix}"] = (
            df[rule] == df[TARGET]
        ).astype(int)
    #list of is_correct names
    is_correct = [name for name in df.keys() if name.startswith("is_correct")]
    print("Precision from value counts")
    #display precision from counts, code from module 3 notebook
    for idx, col in enumerate(is_correct):
        #define precision for predictions on a test dataset
        rule = rules[idx]
        rule_correct = col
        rule_filter = ((df.split == "test") & (df[rule] == 1))

        print(f"rule:{rule}, rule_correct:{rule_correct}")
        print("Counts:",df[rule_filter][rule_correct].value_counts())
        print("--")
        print("Ratio:",(df[rule_filter][rule_correct].value_counts() / len(df[rule_filter])).round(3))

        print("------")
    print("\nPrecision from tp, fp and precision_score")
    #precision using tp, fp. Code from module 3 notebook
    for rule in rules:
        target = df[df["split"] == "test"][TARGET]
        pred = df[df["split"] == "test"][rule]

        tp = ((pred == 1) & (pred == target)).sum()
        fp = ((pred == 1) & (pred != target)).sum()

        #using tp, fp
        precision = tp / (tp + fp)

        print(f"\nRule:{rule} \nprecision (tp, fp):{np.round(precision, 3)}")

#fill missing values
def replace_fill_missing(df:pd.DataFrame):
    #tunr inf to nan either cae +/-
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #fill na with 0
    df.fillna(0, inplace=True)
    return df

#select features to use with models and make it ready
def get_model_data(df: pd.DataFrame, dummy: list[str]) -> list[str]:
    #select desired features
    #growth = [name for name in df.keys() if name.startswith("growth")]
    indicator_prefix = momentum_indicators + volume_indicators + volatility_indicators + cycle_indicators + price_indicators
    indicators = [
        name for name in df.keys() if any(
            name.startswith(prefix) for prefix in indicator_prefix
        )
    ]
    #did not use growth features for training to prevent leakage
    #features = growth + indicators + dummy + [TARGET]
    selected_features = indicators + dummy + [TARGET]
    #return list of selected features
    return selected_features

#define model splits using selected features
def get_model_splits(
        df: pd.DataFrame, features: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #prepare data for model
    train_df = df[df.split.isin(["train", "val"])].copy()
    test_df = df[df.split.isin(["test"])].copy()
    #define splits
    train = train_df[features].copy()
    test = test_df[features].copy()
    #all splits
    all_splits = df[features].copy()
    #return each defined split
    return train, test, all_splits

#get a binary threshold from model predicted probabilities, similar to code from module 3
def get_bin_threshold(y_true, y_pred, only_even=False) -> pd.DataFrame:
    scores = []

    if only_even==False:
       #[0, 0.01, 0.02, ...0.99,1.0]
       thresholds = np.linspace(0, 1, 101)
    else:
       #[0, 0.02, 0.04,  ...0.98,1.0]
       thresholds = np.linspace(0, 1, 51)

    for t in thresholds:
        actual_positive = (y_true == 1)
        actual_negative = (y_true == 0)
        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()
        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        if tp + fp > 0:
          precision = tp / (tp + fp)

        if tp + fn > 0:
          recall = tp / (tp + fn)

        if precision+recall > 0:
          f1_score = 2*precision*recall / (precision+recall)

        accuracy = (tp+tn) / (tp+tn+fp+fn)

        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

    columns = [
       "threshold", "tp", "fp", "fn", "tn",
       "precision", "recall", "accuracy", "f1_score"
    ]
    
    df_scores = pd.DataFrame(scores, columns=columns)
    df_scores["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores

#define new rules using probabilities from DecisionTreeClassifier
def get_proba_hand_rules(
        dt_clf: DecisionTreeClassifier, rf_clf: RandomForestClassifier, 
        x_all: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
    #get proba on entire dataset
    y_pred_prob_all_dtc = dt_clf.predict_proba(x_all)
    y_pred_prob_all_dtc_pos_class = [k[1] for k in y_pred_prob_all_dtc]
    y_pred_prob_all_dtc_pos_class_arr = np.array(y_pred_prob_all_dtc_pos_class)
    #define new rules from probabilities and thresholds
    df["prob_4_dtc"] = y_pred_prob_all_dtc_pos_class_arr
    df["rl_4_dtc_prob_42"] = (
        y_pred_prob_all_dtc_pos_class_arr > 0.42
    ).astype(int)

    df["prob_dtc_5"] = y_pred_prob_all_dtc_pos_class_arr
    df["rl_5_dtc_prob_62"] = (
        y_pred_prob_all_dtc_pos_class_arr > 0.62
    ).astype(int)

    #get proba on entire dataset
    y_pred_prob_all_rfc = rf_clf.predict_proba(x_all)
    y_pred_prob_all_rfc_pos_class = [k[1] for k in y_pred_prob_all_rfc]
    y_pred_prob_all_rfc_pos_class_arr = np.array(y_pred_prob_all_rfc_pos_class)
    #define new rules from probabilities and thresholds
    df["prob_rfc_6"] = y_pred_prob_all_rfc_pos_class_arr
    df["rl_6_rfc_prob_4_6"] = (
        (y_pred_prob_all_rfc_pos_class_arr > 0.4) & (y_pred_prob_all_rfc_pos_class_arr < 0.62)
    ).astype(int)

    df["prob_rfc_7"] = y_pred_prob_all_rfc_pos_class_arr
    df["rl_7_rfc_prob_6_7"] = (
        (y_pred_prob_all_rfc_pos_class_arr > 0.6) & (y_pred_prob_all_rfc_pos_class_arr < 0.82)
    ).astype(int)

    #return df with added rules
    return df