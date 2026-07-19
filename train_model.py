"""
Exoplanet Hunting Dataset — Model Training
======================================================

Produces processed_exoTrain.csv and processed_exoTest.csv in DATA_DIR,
each with engineered features + the LABEL column.
"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier
import joblib
import argparse

def train_model(
        data_dir,
        train_fn,
        test_fn,
        output_dir,
        output_prefix
):
    # Config — edit these paths if your files are elsewhere
    TRAIN_PATH = os.path.join(data_dir, train_fn)
    TEST_PATH = os.path.join(data_dir, test_fn)
    MODEL_OUTPUT_FN = f"{output_prefix}_exoplanet_model.joblib" if output_prefix else "exoplanet_model.joblib"
    MODEL_OUTPUT_PATH = os.path.join(output_dir, MODEL_OUTPUT_FN)

    # 1. Load Processed Data
    print("=" * 60)
    print("1. LOADING PROCESSED DATA")
    print("=" * 60)
    
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    # Remap LABEL: 1 (no planet) -> 0, 2 (planet) -> 1
    train["target"] = train["LABEL"].map({1: 0, 2: 1})
    test["target"] = test["LABEL"].map({1: 0, 2: 1})
    
    feature_cols = [c for c in train.columns if c not in ("LABEL", "target")]
    
    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values
    
    print(f"Features used: {feature_cols}")
    print(f"Train positives: {y_train.sum()} / {len(y_train)}")
    print(f"Test positives:  {y_test.sum()} / {len(y_test)}")

    # 2. Cross-Validated Evaluation on Train — Random Forest vs XGBoost
    # We do this BEFORE touching the test set at all. With only 37 positives
    # in train, a single train/test split is noisy — stratified k-fold CV
    # gives a steadier read on how each model performs on unseen data.
    # Both candidates use the SAME folds, so the comparison is fair.
    print("\n" + "=" * 60)
    print("2. CROSS-VALIDATED EVALUATION — comparing candidate models")
    print("=" * 60)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # XGBoost's imbalance lever

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,          # shallow on purpose — few positives, easy to overfit
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name, candidate in candidates.items():
        print(f"\n--- {name} ---")
        preds = cross_val_predict(candidate, X_train, y_train, cv=cv, method="predict")
        probs = cross_val_predict(candidate, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    
        print(classification_report(y_train, preds, target_names=["no_planet", "planet"]))
        print("Confusion matrix:")
        print(confusion_matrix(y_train, preds))
        auc = roc_auc_score(y_train, probs)
        print(f"ROC-AUC: {auc:.4f}")
    
        cv_results[name] = {"preds": preds, "probs": probs, "auc": auc}
    
    # Pick the winner by ROC-AUC (feel free to weight recall more heavily
    # instead, given that missing a real planet is the costlier error)
    best_name = max(cv_results, key=lambda k: cv_results[k]["auc"])
    print(f"\n>>> Best model by CV ROC-AUC: {best_name} <<<")
    
    model = candidates[best_name]
    
    # 3. Fit Final (Winning) Model on All of Train
    print("\n" + "=" * 60)
    print(f"3. FITTING FINAL MODEL ({best_name}) ON FULL TRAINING SET")
    print("=" * 60)
    
    model.fit(X_train, y_train)
    print("Model fitted.")
    
    # Feature importances - useful for your README's "design decisions" section
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    # 4. Evaluate on Held-Out Test Set
    # Remember: only 5 positives here, so treat these numbers as a rough
    # check, not a precise measurement. The CV numbers above are more
    # trustworthy for understanding real-world performance.
    print("\n" + "=" * 60)
    print("4. EVALUATION ON HELD-OUT TEST SET (small sample, interpret with care)")
    print("=" * 60)
    
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    
    print("Test set classification report:")
    print(classification_report(y_test, test_preds, target_names=["no_planet", "planet"]))
    
    print("Test set confusion matrix:")
    print(confusion_matrix(y_test, test_preds))

    # 5. SAVE THE MODEL
    print("\n" + "=" * 60)
    print("5. SAVING MODEL")
    print("=" * 60)
    
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
    }, MODEL_OUTPUT_PATH)
    print(f"Saved -> {MODEL_OUTPUT_PATH}")
    print("\nThis .joblib file bundles the fitted model AND the exact feature")
    print("column order it expects. Load both together later in your API so")
    print("you never accidentally feed it columns in the wrong order.")

if __name__ == '__main__':
    '''
    python train_model.py \
        --data_dir ../data \
        --train_fn processed_exoTrain.csv \
        --test_fn processed_exoTest.csv \
        --output_dir models
    '''

    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, help='input directory where files are', required=False, default='.')
    parser.add_argument('--train_fn', type=str, help='input train file', required=True)
    parser.add_argument('--test_fn', type=str, help='input test file', required=True)
    parser.add_argument('--output_dir', type=str, help='output directory where model is saved', required=False, default='.')
    parser.add_argument('--output_prefix', type=str, help='output file prefix', required=False)

    args = parser.parse_args()

    train_model(
        
        data_dir=args.data_dir,
        train_fn=args.train_fn,
        test_fn=args.test_fn,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix
    )