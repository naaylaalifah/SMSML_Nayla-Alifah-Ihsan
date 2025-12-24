import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.sklearn.autolog()

def main():
    df = pd.read_csv("winequality_red_preprocessing.csv")

    X = df.drop("quality_label", axis=1)
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    with mlflow.start_run(run_name="rf_basic"):
        model.fit(X_train, y_train)

if __name__ == "__main__":
    main()