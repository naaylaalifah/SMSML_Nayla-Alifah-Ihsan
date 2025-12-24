import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    df = pd.read_csv("winequality_red_preprocessing.csv")

    X = df.drop("quality_label", axis=1)
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    with mlflow.start_run(run_name="rf_tuning_adv"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })
        fi.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()