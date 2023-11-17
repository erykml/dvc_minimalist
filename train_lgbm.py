import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from dvclive import Live
from dvclive.lgbm import DVCLiveCallback

# set the params
train_params = {
    "n_estimators": 10,
    "max_depth": 10,
}

# load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Time"]).copy()
y = X.pop("Class")

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with Live(save_dvc_exp=True) as live:
    # fit-predict
    model = LGBMClassifier(random_state=42, **train_params)
    model.fit(X_train, y_train, callbacks=[DVCLiveCallback(live=live)])
    y_pred = model.predict(X_test)

    # evaluate
    live.log_metric("recall", recall_score(y_test, y_pred))
    live.log_metric("precision", precision_score(y_test, y_pred))
    live.log_metric("f1_score", f1_score(y_test, y_pred))
