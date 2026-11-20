from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#model used params
DTC_DEPTH = 2
RFC_N_EST = 10
RFC_DEPTH = 10

#instantiate DecisionTreeClassifier with specified params
def get_dtc_model() -> DecisionTreeClassifier:
    #instantiate model to use later
    dtc = DecisionTreeClassifier(max_depth=DTC_DEPTH, random_state=42)
    return dtc

#instantiate RandomForestClassifier with specified params
def get_rfc_model() -> RandomForestClassifier:
    #instantiate model to use later
    rfc = RandomForestClassifier(
        n_estimators=RFC_N_EST, max_depth=RFC_DEPTH, random_state=42, n_jobs=1
    )
    return rfc