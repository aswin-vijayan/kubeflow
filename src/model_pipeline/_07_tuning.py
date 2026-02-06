import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score


def tuning_data(train_path: str, test_path: str, preprocessor_path: str) -> dict:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=['Attrition'])
    y_train = df_train['Attrition']

    X_test = df_test.drop(columns=['Attrition'])
    y_test = df_test['Attrition']

    # load preprocessor
    preprocessor = joblib.load(preprocessor_path)


    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # set parameters
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga'],
        'model__l1_ratio': [0],     # equivalent to L2
        'model__max_iter': [1000]
    }

    # set cv
    strat_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # use gridserachcv for tuning model
    grid = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=strat_cv, 
        scoring='recall',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # get best model and save in models/
    tuned_model = grid.best_estimator_

    best_parameters = grid.best_params_
        
    # predict the output with tuned_model
    y_pred = tuned_model.predict(X_test)

    # tuned model evaluation
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "train_score": tuned_model.score(X_train, y_train),
        "test_score": tuned_model.score(X_test, y_test)
    }        


    overall_parameters = {
        **best_parameters,
        **metrics,
    }

    print('parameters: ', overall_parameters)

    return overall_parameters



if __name__ == "__main__":
    import os
    from pathlib import Path
    import json

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    TRAIN_PATH =  DATASET_PATH / "06_preprocess_train_df.csv"
    TEST_PATH = DATASET_PATH / "06_preprocess_test_df.csv"

    ARTIFACTS_PATH = BASE_DIR / "artifacts" / "model_v1"
    PREPROCESSOR_PATH = ARTIFACTS_PATH / "preprocessor.pkl"
    TUNING_METADATA = ARTIFACTS_PATH / "tuning_metadata.json"

    overall_parameters = tuning_data(TRAIN_PATH, TEST_PATH, PREPROCESSOR_PATH)

    with open(TUNING_METADATA, 'w') as f:
        json.dump(overall_parameters, f)
