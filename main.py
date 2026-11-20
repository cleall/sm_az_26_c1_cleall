import pandas as pd
import preprocess.preprocess as prep
import model.model as model
import simulation.simulation as sim

import sklearn, sys, numpy

#list ticker references
ticker_list = ["FMTY14.MX"]
#diplay format
SPACER = "\n---- ---- ---- ----\n"

def main():
    #download data
    print("Downloading ticker data:", ticker_list, end=SPACER)
    data = prep.get_ticker_data(tickers=ticker_list)
    #clean data
    print("Cleaning downloaded data ...", end=SPACER)
    clean_df = data.copy()
    fmty_df = prep.clean_dataframe(clean_df)
    #technical indicator and to predict definition
    print("Creating indicators and to predict features ...", end=SPACER)
    fmty_df = prep.get_technical_indicators(df=fmty_df)
    #get dummies and concat df with them
    print("Getting dummies from data ...", end=SPACER)
    dummies = prep.get_dummies(fmty_df)
    dummies_columns = dummies.columns.to_list()
    #print(dummies_columns, len(dummies_columns))
    fmty_w_dms = pd.concat([fmty_df, dummies], axis=1)
    #print(fmty_w_dms.columns.to_list(), fmty_w_dms.info(), end=SPACER)
    #correlation analysis as in hw
    print("Performing corelation analysis ...", end=SPACER)
    corr_df = prep.get_corr_analysis(fmty_w_dms)
    print(corr_df.head())
    print("month_wom correlation is higher in transition from Q2 to Q3", end=SPACER)
    #create data splits
    print("Creating data splits ...", end=SPACER)
    fmty_w_dms["split"] = prep.get_splits(
        fmty_w_dms,
        min_date = fmty_w_dms.date.min(),
        max_date = fmty_w_dms.date.max()
    )
    print("Splits proportion:", end=SPACER)
    #review split proportions
    print(fmty_w_dms["split"].value_counts() / len(fmty_w_dms))
    print(fmty_w_dms.groupby(["split"])["date"].agg({"min", "max", "count"}), end=SPACER)
    #hand rules definition, my notebook and py files flow changes here idk why
    fmty_w_dms = prep.get_hand_rules(fmty_w_dms)
    #display rules precision
    print("Show hand rules precision", end=SPACER)
    prep.show_hand_rules_precision(fmty_w_dms)
    #get features for modeling
    print("Preparing data for modeling ...", end=SPACER)
    features = prep.get_model_data(fmty_w_dms, dummies_columns)
    #print(features, len(features), end=SPACER)
    #define model splits
    x_train, x_test, x_all = prep.get_model_splits(fmty_w_dms, features)
    print(f"len: x_train {x_train.shape}, x_test {x_test.shape}, x_all: x_all {x_all.shape}", end=SPACER)
    #replace missing
    x_train = prep.replace_fill_missing(x_train)
    x_test = prep.replace_fill_missing(x_test)
    x_all = prep.replace_fill_missing(x_all)
    #isolate target
    y_train = x_train[prep.TARGET]
    #y_test = x_test[prep.TARGET]
    #y_all = x_all[prep.TARGET]
    #remove target from features
    del x_train[prep.TARGET]
    del x_test[prep.TARGET]
    del x_all[prep.TARGET]
    #instantiate/fit model with already defined best estimators from notebook
    #DecisionTreeClassifier
    print("Instantiate and fit models with modeling data ...", end=SPACER)
    dt_clf = model.get_dtc_model()
    dt_clf.fit(x_train, y_train)
    #RandomForestClassifier
    rf_clf = model.get_rfc_model()
    rf_clf.fit(x_train, y_train)
    #get probabilities from DecisionTreeClassifier on test
    y_pred_prob_dtc = dt_clf.predict_proba(x_test)
    y_pred_pos_class_dtc = [prob[1] for prob in y_pred_prob_dtc]
    y_pred_pos_class_dtc_df = pd.DataFrame(y_pred_pos_class_dtc, columns=['prob_pos_class'])
    print("Descriptive statistics for DecisionTreeClasifier predicted probabilities on test", end=SPACER)
    print(y_pred_pos_class_dtc_df.describe().T)
    #get probabilities from RandomForestClassifier on test
    y_pred_prob_rfc = rf_clf.predict_proba(x_test)
    y_pred_pos_class_rfc = [prob[1] for prob in y_pred_prob_rfc]
    y_pred_pos_class_rfc_df = pd.DataFrame(y_pred_pos_class_rfc, columns=["prob_pos_class"])
    print("Descriptive statistics for RandomForestClassifier predicted probabilities on test", end=SPACER)
    print(y_pred_pos_class_rfc_df.describe().T, end=SPACER)
    #get binary threshold for probabilities from DecisionTreeClassifier, only for review purposes
    #df_scores_dtc = prep.get_bin_threshold(y_test, y_pred_pos_class_dtc, only_even=True)
    #get binary threshold for probabilities from RandomForestClassifier
    #df_scores_rfc = prep.get_bin_threshold(y_test, y_pred_pos_class_rfc, only_even=True)
    #use a copy for probabilities
    fmty_w_dms_prob = fmty_w_dms.copy()
    #define new rules after reviewing probabilities for DecisionTreeClassifier, RandomForestClassifier
    fmty_w_dms_prob = prep.get_proba_hand_rules(
        dt_clf=dt_clf, rf_clf=rf_clf, x_all=x_all, df=fmty_w_dms_prob
    )
    #display rules precision for probabilities
    print("Show hand rules using probabilities precision", end=SPACER)
    prep.show_hand_rules_precision(fmty_w_dms_prob)
    #show trading simulation
    print("Running simulation ...", end=SPACER)
    sim_rslts_df = sim.simulate_hand_rules(df=fmty_w_dms_prob)
    print(sim_rslts_df.head(10), end=SPACER)
    sim_rslts_df["sim_1_growth_capital_2y"] = (sim_rslts_df.total_net_rev + sim_rslts_df.inv_capital) / sim_rslts_df.inv_capital
    print(sim_rslts_df.sort_values(by="estimation_cagr", ascending=False)[["rule", "sim_1_growth_capital_2y", "estimation_cagr"]])

if __name__ == "__main__":
    main()