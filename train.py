import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from dvclive import Live

# set the params
train_params = {
    "n_estimators": 10,
    "max_depth": 5,
}

# load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Time"]).copy()
y = X.pop("Class")

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# fit-predict
model = RandomForestClassifier(random_state=42, **train_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# evaluate
with Live(save_dvc_exp=True) as live:
    for param_name, param_value in train_params.items():
        live.log_param(param_name, param_value)
    live.log_metric("recall", recall_score(y_test, y_pred))
    live.log_metric("precision", precision_score(y_test, y_pred))
    live.log_metric("f1_score", f1_score(y_test, y_pred))
