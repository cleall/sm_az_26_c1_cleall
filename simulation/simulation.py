import pandas as pd
import numpy as np
#constants to customize simulation
from preprocess.preprocess import (
    TARGET_SRC, TRADING_DAYS
)

#years for CAGR
YEARS_CAGR = 2
#the amount invested starts in 20 mxn min 1 buy, reasonable to start
DLY_INVST_AMT = 20
#0.25%
FEE = 0.0025
#IVA 16%, calculated on the fee amount obtained e.g. (100 * 0.0025) -> this is the fee amount
IVA = 0.16

def simulate_hand_rules(df: pd.DataFrame) -> pd.DataFrame:
    #store result here
    sim_1_rslts = []
    #get rule names
    rules = [pred_name for pred_name in df.keys() if pred_name.startswith("rl")]
    #results for rules (manual/models)
    for rule in rules:
        print(f"Simulation for rule: {rule} in progress ...")
        test_rule_count = len(df[
            (df.split=="test") & (df[rule]==1) & (df[TARGET_SRC].notna())
        ])
        test_count = len(df[(df.split=="test")])
        print(f"Count times of investment {test_rule_count} out of {test_count} TEST records")

        #rule prefixes rl_1, rl_2 ... rl_n
        name = rule.split("_")[0:2]
        rl_prefix = "_".join(name)

        #agg results
        df["sim_1_"+rl_prefix+"_gross_rev"] = df[rule] * DLY_INVST_AMT * (df[TARGET_SRC] - 1)
        df["sim_1_"+rl_prefix+"_fee"] = -df[rule] * DLY_INVST_AMT * FEE
        df["sim_1_"+rl_prefix+"_iva"] = df["sim_1_"+rl_prefix+"_fee"] * IVA
        df["sim_1_"+rl_prefix+"_net_rev"] = df["sim_1_"+rl_prefix+"_gross_rev"] + df["sim_1_"+rl_prefix+"_fee"] + df["sim_1_"+rl_prefix+"_iva"]

        #use test observations where rule is true
        test_rule_pos = (df.split=="test") & (df[rule]==1) & (df[TARGET_SRC].notna())
        #count number of investments
        inv_count = len(df[test_rule_pos])
        #get rule totals where rule is true in each agg result
        total_gross_rev = df[test_rule_pos]["sim_1_"+rl_prefix+"_gross_rev"].sum()
        total_fees = df[test_rule_pos]["sim_1_"+rl_prefix+"_fee"].sum()
        total_iva = df[test_rule_pos]["sim_1_"+rl_prefix+"_iva"].sum()
        total_net_rev = df[test_rule_pos]["sim_1_"+rl_prefix+"_net_rev"].sum()

        #get total fees when gross_rev is ot negligible to data scale
        scale = df["sim_1_"+rl_prefix+"_net_rev"].abs().max()
        #10 orders of magnitude below the largest value
        threshold = 1e-10 * scale
        if abs(total_net_rev) > threshold:
            fees_percentage = -(total_fees + total_iva) / total_net_rev
        else:
            fees_percentage = None

        #get net revenue when investment were made
        if inv_count > 0:
            avg_net_rev = total_net_rev / inv_count
        else:
            avg_net_rev = None

        #approx capital required and CAGR Calculation
        inv_count_dly_df = pd.DataFrame(df[test_rule_pos].groupby("date")[rule].count())
        avg_inv_per_day = inv_count_dly_df[rule].mean()
        #75% case how many investments per day?
        p_75_inv_per_day = inv_count_dly_df[rule].quantile(0.75)
        #inv_count_dly_df[rule].mean()
        #21 (trading) days in a row with positive rule
        inv_capital = DLY_INVST_AMT * TRADING_DAYS * p_75_inv_per_day
        #remember past performance does not guarantee future results
        #cagr here is only an estimation in n=2 (~ 2 years, 1.8 real)
        estimation_cagr = ((inv_capital + total_net_rev)/inv_capital)**(1/YEARS_CAGR)

        # append to DF
        sim_1_rslts.append(
            (rule, inv_count, total_gross_rev, total_fees, total_iva, total_net_rev,
            fees_percentage, avg_net_rev, avg_inv_per_day, inv_capital, estimation_cagr)
        )

        #output for all rules with positive predictions
        columns = [
            "sim_1_"+rl_prefix+"_gross_rev", "sim_1_"+rl_prefix+"_fee",
            "sim_1_"+rl_prefix+"_iva", "sim_1_"+rl_prefix+"_net_rev"
        ]

        if inv_count > 1:
            print(f"Financial Result:\n{df[test_rule_pos][columns].sum()}")
            print("--[Rounding to 3 decimals only for display purposes]--")
            print(f"  Count Investments in {YEARS_CAGR} years (on TEST): {inv_count}")
            print(f"  Gross Revenue: MX${np.round(total_gross_rev, 3)}")
            print(f"  Fees (0.25% for buy+sell): MX${-np.round(total_fees, 3)}")
            print(f"  IVA (16% on Fee): MX${-np.round(total_iva, 3)}")
            print(f"  Net Revenue: MX${np.round(total_net_rev, 3)}")
            print(f"  Fees are {-np.round(100.0 * (total_fees + total_iva) / total_gross_rev, 3)}% of Gross Revenue")
            print(f"  Capital Required : MX${inv_capital} (Vbegin)")
            print(f"  Final value (Vbegin + Net_revenue) : MX${np.round(inv_capital + total_net_rev, 3)} (Vfinal)\n")

            print(f"Average CAGR on TEST ({YEARS_CAGR} years): {np.round(estimation_cagr, 3)}, or {np.round(100.0 * (estimation_cagr - 1), 1)}%")
            print("Average daily stats:")
            print(f"  Average net revenue per investment: MX${total_net_rev / inv_count}")
            print(f"  Average investments per day: {int(np.round(avg_inv_per_day))}")
            print(f"  Q75 investments per day: {int(np.round(p_75_inv_per_day))}")
            print("=============================================")

    #select columns to build df
    columns_simulation = [
        "rule", "inv_count", "total_gross_rev", "total_fees", "total_iva", "total_net_rev",
        "fees_percentage", "avg_net_rev", "avg_inv_per_day", "inv_capital", "estimation_cagr"
    ]
    #save to df
    sim_1_rslts_df = pd.DataFrame(sim_1_rslts, columns=columns_simulation)
    #return df with results
    return sim_1_rslts_df