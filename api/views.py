from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
from django.conf import settings
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
 
 


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import root_mean_squared_error



class BaselineModel:
    def __init__(self, task_type):
        self.task_type = task_type

    def get(self):
        if self.task_type == "Regression":
            return LinearRegression()
        elif self.task_type == "Classification":
            return LogisticRegression(max_iter=1000)


class TrainTestSplitter:
    def __init__(self, task_type, test_size=0.2, random_state=42):
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        stratify = y if self.task_type == "Classification" else None
        return train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state,
                                stratify=stratify)


class FeatureSelector:
    def __init__(self, task_type):
        self.task_type = task_type

    def select(self, X_train, y_train):
        if self.task_type == "Regression":
            selector_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            selector_model = RandomForestClassifier(n_estimators=100, random_state=42)

        selector_model.fit(X_train, y_train)
        selector = SelectFromModel(selector_model, prefit=True, threshold="median")
        return selector


class ModelRegistry:
    def __init__(self, task_type):
        self.task_type = task_type

    def get_models(self):
        if self.task_type == "Regression":
            return {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(random_state=42),
                "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
                "SVR": SVR(),
            }
        elif self.task_type == "Classification":
            return {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
                "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True),
            }


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.metrics import root_mean_squared_error
import numpy as np

class ModelEvaluator:
    def __init__(self, task_type):
        self.task_type = task_type

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        if self.task_type == "Regression":
            return {
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": root_mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }

        elif self.task_type == "Classification":
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1": f1_score(y_test, y_pred, average="weighted")
            }

            # ROC-AUC handling
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)

                # Binary classification
                if y_prob.shape[1] == 2:
                    metrics["ROC_AUC"] = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    # Multiclass
                    metrics["ROC_AUC"] = roc_auc_score(y_test, y_prob, multi_class="ovr")
            else:
                metrics["ROC_AUC"] = None

            return metrics


class AutoModelingSystem:
    def __init__(self, X, y, problem):
        self.X = X
        self.y = y
        if isinstance(problem, tuple):
            problem = problem[0]
        self.problem = problem
        self.task_type = problem["task_type"]

    def run(self):
        results = {}

        # Step 1: Train-test split
        splitter = TrainTestSplitter(self.task_type)
        X_train, X_test, y_train, y_test = splitter.split(self.X, self.y)

        # Step 2: Baseline model
        baseline_model = BaselineModel(self.task_type).get()
        baseline_model.fit(X_train, y_train)
        evaluator = ModelEvaluator(self.task_type)
        baseline_metrics = evaluator.evaluate(baseline_model, X_test, y_test)

        results["Baseline"] = {
            "model": baseline_model,
            "metrics": baseline_metrics
        }

        # Step 3: Feature selection
        selector = FeatureSelector(self.task_type).select(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        # Step 4: Model training
        registry = ModelRegistry(self.task_type)
        models = registry.get_models()

        model_results = {}

        for name, model in models.items():
            model.fit(X_train_sel, y_train)
            metrics = evaluator.evaluate(model, X_test_sel, y_test)
            model_results[name] = {
                "model": model,
                "metrics": metrics
            }

        results["Models"] = model_results

        # Step 5: Select best model
        best_model_name, best_model_info = self.select_best_model(model_results)
        results["BestModel"] = {
            "name": best_model_name,
            "model": best_model_info["model"],
            "metrics": best_model_info["metrics"]
        }

        return results

    def select_best_model(self, model_results):
        if self.task_type == "Regression":
            # Minimize RMSE
            best = min(model_results.items(), key=lambda x: x[1]["metrics"]["RMSE"])
        else:
            # Maximize Accuracy
            best = max(model_results.items(), key=lambda x: x[1]["metrics"]["Accuracy"])
        return best



class DatasetProfiler:
    def __init__(self, df, problem):
        self.df = df
        self.problem = problem

    def profile(self):
        profile = {
            "target": self.problem["target"],
            "inputs": self.problem["inputs"],
            "columns": {}
        }

        for col in self.problem["inputs"] + [self.problem["target"]]:
            series = self.df[col]
            profile["columns"][col] = {
                "dtype": str(series.dtype),
                "missing_pct": series.isnull().mean(),
                "unique_values": series.nunique(),
                "sample_values": series.dropna().unique()[:5].tolist(),
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "is_text": pd.api.types.is_string_dtype(series),
                "is_datetime": pd.api.types.is_datetime64_any_dtype(series)
            }

        return profile


class DataCleaner:
    def __init__(self, df, problem):
        self.df = df.copy()
        self.problem = problem

    def clean(self):
        # Drop duplicates
        self.df = self.df.drop_duplicates()

        # Drop rows with missing target
        self.df = self.df[self.df[self.problem["target"]].notnull()]

        # Cap extreme outliers in numeric inputs
        for col in self.problem["inputs"]:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                q1 = self.df[col].quantile(0.01)
                q99 = self.df[col].quantile(0.99)
                self.df[col] = self.df[col].clip(q1, q99)

        return self.df


class ColumnTypeInferencer:
    def __init__(self, profile):
        self.profile = profile

    def infer(self):
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        datetime_cols = []

        for col, info in self.profile["columns"].items():
            if col == self.profile["target"]:
                continue

            if info["is_datetime"]:
                datetime_cols.append(col)
            elif info["is_numeric"]:
                numeric_cols.append(col)
            elif info["is_text"]:
                # Distinguish categorical vs free text
                if info["unique_values"] <= 30:
                    categorical_cols.append(col)
                else:
                    text_cols.append(col)
            else:
                categorical_cols.append(col)

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "text": text_cols,
            "datetime": datetime_cols
        }


class PreprocessingStrategySelector:
    def __init__(self, df, profile, column_types):
        self.df = df
        self.profile = profile
        self.column_types = column_types

    def select(self):
        strategies = {}

        # Numeric strategy
        strategies["numeric"] = {
            "imputer": "median",
            "scaler": "robust",
            "transform": None
        }

        # Categorical strategy
        strategies["categorical"] = {
            "imputer": "most_frequent",
            "encoder": "onehot"
        }

        # Text strategy
        strategies["text"] = {
            "vectorizer": "tfidf"
        }

        # Datetime strategy
        strategies["datetime"] = {
            "feature_engineering": "extract_parts"
        }

        # Target transformation (regression only)
        target_col = self.profile["target"]
        if self.profile["columns"][target_col]["is_numeric"]:
            skew = self.df[target_col].skew()
            if abs(skew) > 1:
                strategies["target_transform"] = "log"
            else:
                strategies["target_transform"] = None

        return strategies


class DateTimeFeatureEngineer:
    def __init__(self, df, datetime_cols):
        self.df = df
        self.datetime_cols = datetime_cols

    def transform(self):
        for col in self.datetime_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            self.df[f"{col}_year"] = self.df[col].dt.year
            self.df[f"{col}_month"] = self.df[col].dt.month
            self.df[f"{col}_day"] = self.df[col].dt.day
            self.df[f"{col}_dayofweek"] = self.df[col].dt.dayofweek
            self.df = self.df.drop(columns=[col])
        return self.df


class PipelineBuilder:
    def __init__(self, column_types, strategies):
        self.column_types = column_types
        self.strategies = strategies

    def build(self):
        transformers = []

        # Numeric pipeline
        if self.column_types["numeric"]:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy=self.strategies["numeric"]["imputer"])),
                ("scaler", RobustScaler() if self.strategies["numeric"]["scaler"] == "robust" else StandardScaler())
            ])
            transformers.append(("num", num_pipeline, self.column_types["numeric"]))

        # Categorical pipeline
        if self.column_types["categorical"]:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy=self.strategies["categorical"]["imputer"])),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            transformers.append(("cat", cat_pipeline, self.column_types["categorical"]))

        # Text pipeline
        if self.column_types["text"]:
            text_pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(max_features=5000))
            ])
            transformers.append(("text", text_pipeline, self.column_types["text"][0]))  # sklearn supports 1 text col

        return ColumnTransformer(transformers)


class TargetTransformer:
    def __init__(self, strategy):
        self.strategy = strategy

    def transform(self, y):
        if self.strategy == "log":
            return np.log1p(y)
        return y

    def inverse_transform(self, y):
        if self.strategy == "log":
            return np.expm1(y)
        return y


class PreprocessingValidator:
    def __init__(self, df, problem, column_types):
        self.df = df
        self.problem = problem
        self.column_types = column_types

    def validate(self):
        report = {
            "target_exists": self.problem["target"] in self.df.columns,
            "all_inputs_exist": all(col in self.df.columns for col in self.problem["inputs"]),
            "no_missing_target": bool(self.df[self.problem["target"]].isnull().sum() == 0),
            "feature_columns_non_empty": any(len(v) > 0 for v in self.column_types.values()),
            "leakage_detected": self.problem["target"] in self.problem["inputs"]
        }

        report["pipeline_safe"] = all([
            report["target_exists"],
            report["all_inputs_exist"],
            report["no_missing_target"],
            report["feature_columns_non_empty"],
            not report["leakage_detected"]
        ])

        return report


class IntelligentPreprocessingSystem:
    def __init__(self, df, problem):
        if isinstance(problem, tuple):
            problem = problem[0]
        self.df = df
        self.problem = problem

    def run(self):
        # Step 1: Clean data
        cleaner = DataCleaner(self.df, self.problem)
        df_clean = cleaner.clean()

        # Step 2: Profile
        profiler = DatasetProfiler(df_clean, self.problem)
        profile = profiler.profile()

        # Step 3: Infer column types
        inferencer = ColumnTypeInferencer(profile)
        column_types = inferencer.infer()

        # Step 4: Feature engineering for datetime
        if column_types["datetime"]:
            dt_engineer = DateTimeFeatureEngineer(df_clean, column_types["datetime"])
            df_clean = dt_engineer.transform()

            # Re-profile after datetime expansion
            profiler = DatasetProfiler(df_clean, self.problem)
            profile = profiler.profile()
            inferencer = ColumnTypeInferencer(profile)
            column_types = inferencer.infer()

        # Step 5: Select preprocessing strategies
        strategy_selector = PreprocessingStrategySelector(df_clean, profile, column_types)
        strategies = strategy_selector.select()

        # Step 6: Build preprocessing pipeline
        pipeline_builder = PipelineBuilder(column_types, strategies)
        preprocessor = pipeline_builder.build()

        # Step 7: Validate
        validator = PreprocessingValidator(df_clean, self.problem, column_types)
        validation_report = validator.validate()

        # Step 8: Prepare X and y
        X = df_clean[self.problem["inputs"]]
        y = df_clean[self.problem["target"]]

        # Step 9: Target transform if needed
        target_transformer = TargetTransformer(strategies.get("target_transform"))
        y_transformed = target_transformer.transform(y)

        return {
            "X": X,
            "y": y_transformed,
            "preprocessor": preprocessor,
            "target_transformer": target_transformer,
            "profile": profile,
            "column_types": column_types,
            "strategies": strategies,
            "validation_report": validation_report,
            "ready_for_modeling": validation_report["pipeline_safe"]
        }


class FrontendResultFormatter:
    def __init__(self, results, X, y, problem):
        self.results = results
        self.X = X
        self.y = y
        self.problem = problem

    def format(self):
        best = self.results["BestModel"]
        models = self.results["Models"]

        summary = self.build_summary(best)
        model_comparison = self.build_model_comparison(models)
        feature_importance = self.build_feature_importance(best["model"])
        training_metrics = self.build_training_metrics(best["model"])

        return {
            "summary": summary,
            "training_metrics": training_metrics,
            "feature_importance": feature_importance,
            "model_comparison": model_comparison,
            "best_model": {
                "name": best["name"],
                "metrics": best["metrics"],
            }
        }

    def build_summary(self, best):
        metrics = best["metrics"]

        return {
            "accuracy": metrics.get("Accuracy"),
            "f1": metrics.get("F1"),
            "loss": metrics.get("Loss", None),  # Optional
            "inference_time_ms": None,  # Can be added later
            "best_model": best["name"]
        }

    def build_model_comparison(self, models):
        comparison = {}
        for name, info in models.items():
            comparison[name] = info["metrics"]
        return comparison

    def build_feature_importance(self, model):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            features = [f"Feature_{i}" for i in range(len(importances))]
            return [
                {"feature": f, "importance": float(i)}
                for f, i in zip(features, importances)
            ]
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            features = [f"Feature_{i}" for i in range(len(coefs))]
            return [
                {"feature": f, "importance": abs(float(c))}
                for f, c in zip(features, coefs)
            ]
        else:
            return []

    def build_training_metrics(self, model):
        # Placeholder — sklearn doesn't expose epoch curves easily
        return {
            "epochs": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_loss": [],
            "val_loss": []
        }

problem =   {
      "problem_id": 1,
      "problem_name": "Classify Iris Species",
      "task_type": "Classification",
      "inputs": [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
      ],
      "target": "Species",
      "business_interpretation": "Automatically identifying iris species based on flower measurements reduces manual classification effort and improves accuracy in botanical research and education."
    }


def make_json_safe(result):
    safe_result = {}

    for key, value in result.items():
        if isinstance(value, dict):
            safe_result[key] = make_json_safe(value)
        else:
            safe_result[key] = str(value)

    return safe_result

def serialize_model_results(results):
    serialized = {}

    # Serialize Baseline
    serialized['Baseline'] = {
        'model': str(results['Baseline']['model'].__class__.__name__),
        'metrics': results['Baseline']['metrics']
    }

    # Serialize Models
    serialized['Models'] = {}
    for name, info in results['Models'].items():
        serialized['Models'][name] = {
            'model': str(info['model'].__class__.__name__),
            'metrics': info['metrics']
        }

    # Serialize BestModel
    serialized['BestModel'] = {
        'name': results['BestModel']['name'],
        'model': str(results['BestModel']['model'].__class__.__name__),
        'metrics': results['BestModel']['metrics']
    }

    return serialized




def profile_dataset(df):
    profile = {}
    profile["n_rows"] = len(df)
    profile["n_columns"] = len(df.columns)
    profile["columns"] = []

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_values": df[col].nunique(),
            "missing_pct": df[col].isnull().mean(),
            "sample_values": df[col].dropna().unique()[:5].tolist()
        }
        profile["columns"].append(col_info)

    return profile
def hello_api(request):
    return JsonResponse({'message': 'Hello, World!'})

def infer_tasks(profile):
    tasks = []

    for col in profile["columns"]:
        if col["dtype"] in ["int64", "float64"] and col["unique_values"] > 10:
            tasks.append({
                "target": col["name"],
                "task_type": "Regression"
            })
        elif col["dtype"] == "object" and col["unique_values"] <= 10:
            tasks.append({
                "target": col["name"],
                "task_type": "Classification"
            })

    if not tasks:
        tasks.append({"task_type": "Clustering"})

    return tasks


def get_target_distribution_data(df_clean, y_transformed, problem):
    target = problem["target"]
    task_type = problem["task_type"]

    raw_target = df_clean[target]

    if task_type == "Regression":
        return {
            "graph_type": "distribution_comparison",
            "target": target,
            "raw_distribution": raw_target.tolist(),
            "processed_distribution": y_transformed.tolist(),
            "bins": 30,
            "description": "Shows how preprocessing transformed the target distribution to improve model learnability."
        }

    elif task_type == "Classification":
        raw_counts = raw_target.value_counts().to_dict()
        return {
            "graph_type": "class_balance",
            "target": target,
            "class_counts": raw_counts,
            "description": "Shows class distribution to assess imbalance and classification feasibility."
        }


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd

def get_feature_target_relationship_data(X, y, column_types, problem, top_k=3):
    task_type = problem["task_type"]
    numeric_features = column_types["numeric"]
    categorical_features = column_types["categorical"]
    text_features = column_types["text"]

    feature_scores = {}

    if task_type == "Regression":
        for col in numeric_features:
            corr = pd.Series(X[col]).corr(pd.Series(y))
            if not pd.isna(corr):
                feature_scores[col] = abs(corr)

    elif task_type == "Classification":
        X_numeric = X[numeric_features] if numeric_features else pd.DataFrame()
        if not X_numeric.empty:
            mi_scores = mutual_info_classif(X_numeric, y, discrete_features=False)
            for col, score in zip(numeric_features, mi_scores):
                feature_scores[col] = score

    # Select top numeric features
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    graphs = []
    for feature, score in top_features:
        if task_type == "Regression":
            graphs.append({
                "feature": feature,
                "target": problem["target"],
                "graph_type": "scatter",
                "x": X[feature].tolist(),
                "y": y.tolist(),
                "strength": score,
                "description": f"Shows relationship between {feature} and target."
            })
        elif task_type == "Classification":
            df_temp = pd.DataFrame({"feature": X[feature], "target": y})
            box_data = {
                str(cls): df_temp[df_temp["target"] == cls]["feature"].tolist()
                for cls in sorted(df_temp["target"].unique())
            }
            graphs.append({
                "feature": feature,
                "target": problem["target"],
                "graph_type": "boxplot",
                "groups": box_data,
                "strength": score,
                "description": f"Shows distribution of {feature} across target classes."
            })

    # 🔥 NEW: Handle text-only datasets gracefully
    if not graphs and text_features:
        graphs.append({
            "feature": text_features[0],
            "target": problem["target"],
            "graph_type": "text_semantic_signal",
            "description": "Text features were transformed using TF-IDF. Semantic signal exists but is represented in high-dimensional vector space.",
            "strength": "High (semantic embedding)"
        })

    return {
        "graph_type": "feature_target_relationships",
        "top_features": graphs,
        "description": "Highlights the strongest relationships between features and the target variable."
    }


def get_preprocessing_impact_data(df_original, df_clean, column_types, strategies):
    impact = {}

    # Missing values
    missing_before = df_original.isnull().mean().mean() * 100
    missing_after = df_clean.isnull().mean().mean() * 100

    # Feature counts
    num_features = len(column_types["numeric"])
    cat_features = len(column_types["categorical"])
    text_features = len(column_types["text"])
    datetime_features = len(column_types["datetime"])

    impact["data_quality"] = {
        "missing_pct_before": round(missing_before, 2),
        "missing_pct_after": round(missing_after, 2)
    }

    impact["feature_composition"] = {
        "numeric": num_features,
        "categorical": cat_features,
        "text": text_features,
        "datetime": datetime_features
    }

    impact["strategies_applied"] = strategies

    return {
        "graph_type": "preprocessing_impact",
        "data_quality": impact["data_quality"],
        "feature_composition": impact["feature_composition"],
        "strategies_applied": impact["strategies_applied"],
        "description": "Summarizes how preprocessing improved data quality and prepared features for modeling."
    }


def get_readiness_summary(validation_report, column_types, strategies):
    readiness = {
        "model_readiness": "High" if validation_report["pipeline_safe"] else "Low",
        "checks": {
            "Target detected": validation_report["target_exists"],
            "Inputs detected": validation_report["all_inputs_exist"],
            "No missing target values": validation_report["no_missing_target"],
            "No data leakage": not validation_report["leakage_detected"],
            "Valid feature set": validation_report["feature_columns_non_empty"]
        },
        "feature_types_detected": column_types,
        "preprocessing_strategies": strategies,
        "summary_message": (
            "Dataset is ready for modeling." if validation_report["pipeline_safe"]
            else "Dataset has issues that must be resolved before modeling."
        )
    }

    return readiness


# Run preprocessing

import json

from google import genai
def infer_ml_tasks_from_csv(csv_path, api_key, model_name='gemini-3-flash-preview'):
    print("successful yahan tak", csv_path, api_key, model_name)

    # 1️⃣ Load dataset
    df = pd.read_csv(csv_path)
    print("okay")
    print(df.head())
    # 2️⃣ Configure Gemini





    client =  genai.Client(api_key=api_key)
    
    # 3️⃣ Create prompt
    prompt = f"""
{profile_dataset(df)}
{infer_tasks(profile_dataset(df))}
You are a data scientist. Given this dataset schema and column samples, infer:

- What each column represents
- Which columns are likely targets
- Which columns are features
- Any temporal structure
- no text only json
- task_type can only be regression or classification nothing else !
- list as many as you think models can be train by seeing this data only supervised models 
- exact column names like given to you and exact target name i gave you
- use exact structure
Example output:
"{{
    "problem_id": 1,
      "problem_name": "Spam SMS Detection",
      "task_type": "Classification",
      "inputs": [
        "Message"
      ],
      "target": "Category",
      "business_interpretation": "Automatically classifying SMS messages as spam or legitimate (ham) helps telecom providers and messaging platforms protect users from fraud, phishing attempts, and unwanted promotional content."

}} and other example are
  {{
    "problem_id": 1,
    "problem_name": "Iris Species Classification",
    "task_type": "Classification",
    "inputs": [
      "SepalLengthCm",
      "SepalWidthCm",
      "PetalLengthCm",
      "PetalWidthCm"
    ],
    "target": "Species",
    "business_interpretation": "Classifying iris flowers into their respective species (Setosa, Versicolor, Virginica) based on morphological measurements allows botanists and researchers to automate species identification and study biological variance."
  }},
  {{
    "problem_id": 2,
    "problem_name": "Petal Width Estimation",
    "task_type": "Regression",
    "inputs": [
      "SepalLengthCm",
      "SepalWidthCm",
      "PetalLengthCm",
      "Species"
    ],
    "target": "PetalWidthCm",
    "business_interpretation": "Predicting petal width based on other floral dimensions and species type helps in understanding the growth patterns and health of the plant when specific measurements are difficult to obtain."
  }},
  {{
    "problem_id": 3,
    "problem_name": "Petal Length Prediction",
    "task_type": "Regression",
    "inputs": [
      "SepalLengthCm",
      "SepalWidthCm",
      "PetalWidthCm",
      "Species"
    ],
    "target": "PetalLengthCm",
    "business_interpretation": "Modeling petal length as a function of sepal size and species helps in phenotypic profiling and plant breeding selection processes."
  }},
 {{
    "problem_id": 4,
    "problem_name": "Sepal Length Prediction",
    "task_type": "Regression",
    "inputs": [
      "SepalWidthCm",
      "PetalLengthCm",
      "PetalWidthCm",
      "Species"
    ],
    "target": "SepalLengthCm",
    "business_interpretation": "Estimating sepal length helps researchers reconstruct the physical profile of a specimen if parts of the flower are damaged or missing in herbarium samples."
  }},
  {{
    "problem_id": 5,
    "problem_name": "Sepal Width Prediction",
    "task_type": "Regression",
    "inputs": [
      "SepalLengthCm",
      "PetalLengthCm",
      "PetalWidthCm",
      "Species"
    ],
    "target": "SepalWidthCm",
    "business_interpretation": "Predicting the width of the sepal provides insights into the structural integrity and symmetry of the flower, which are often markers for specific environmental adaptations."
  }}
"

"""


    
    # 4️⃣ Call Gemini model
    # response = client.models.generate_content(
    #     model=model_name,
    #     contents=prompt
    # )
    # print("Gemini response:", response.text)
    # 5️⃣ Convert response text to JSON
    response = {}
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        print("Error parsing JSON from Gemini:", e)
        print("Raw response:", response.text)
        return None
    
    return data



@csrf_exempt
def upload_csv(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)

    uploaded_file = request.FILES['file']
    file_name = uploaded_file.name.lower()

        # Load file
    if file_name.endswith('.csv'):
     df = pd.read_csv(uploaded_file)
    elif file_name.endswith(('.xls', '.xlsx')):
     df = pd.read_excel(uploaded_file)
    print(uploaded_file)

    try:
        # Read CSV directly from memory
        print(df.head())
        result = profile_dataset(df)   # your ML / profiling logic

        GOOGLE_API_KEY = "AIzaSyC2KPF8SOsBoYPkVJVEdqruGsH26EZesgU"
        # csv_file = "/content/global_supply_chain_disruption_v1.csv"
        client =  genai.Client(api_key=GOOGLE_API_KEY)
    
    # 3️⃣ Create prompt
        prompt = f"""
{profile_dataset(df)}
{infer_tasks(profile_dataset(df))}
You are a data scientist. Given this dataset schema and column samples, infer:

- What each column represents
- Which columns are likely targets
- Which columns are features
- Any temporal structure
- no text only json
- task_type can only be regression or classification nothing else !
- list as many as you think models can be train by seeing this data only supervised models 
- exact column names like given to you and exact target name i gave you
- use exact structure
Example output:
"{{
    "problem_id": 1,
      "problem_name": "Spam SMS Detection",
      "task_type": "Classification",
      "inputs": [
        "Message"
      ],
      "target": "Category",
      "business_interpretation": "Automatically classifying SMS messages as spam or legitimate (ham) helps telecom providers and messaging platforms protect users from fraud, phishing attempts, and unwanted promotional content."

}} and other example are
  {{
    "problem_id": 1,
    "problem_name": "Iris Species Classification",
    "task_type": "Classification",
    "inputs": [
      "SepalLengthCm",
      "SepalWidthCm",
      "PetalLengthCm",
      "PetalWidthCm"
    ],
    "target": "Species",
    "business_interpretation": "Classifying iris flowers into their respective species (Setosa, Versicolor, Virginica) based on morphological measurements allows botanists and researchers to automate species identification and study biological variance."
  }},
  {{
    "problem_id": 2,
    "problem_name": "Petal Width Estimation",
    "task_type": "Regression",
    "inputs": [
      "SepalLengthCm",
      "SepalWidthCm",
      "PetalLengthCm",
      "Species"
    ],
    "target": "PetalWidthCm",
    "business_interpretation": "Predicting petal width based on other floral dimensions and species type helps in understanding the growth patterns and health of the plant when specific measurements are difficult to obtain."
  }},
  {{
    "problem_id": 3,
    "problem_name": "Petal Length Prediction",
    "task_type": "Regression",
    "inputs": [
      "SepalLengthCm",
      "SepalWidthCm",
      "PetalWidthCm",
      "Species"
    ],
    "target": "PetalLengthCm",
    "business_interpretation": "Modeling petal length as a function of sepal size and species helps in phenotypic profiling and plant breeding selection processes."
  }},
 {{
    "problem_id": 4,
    "problem_name": "Sepal Length Prediction",
    "task_type": "Regression",
    "inputs": [
      "SepalWidthCm",
      "PetalLengthCm",
      "PetalWidthCm",
      "Species"
    ],
    "target": "SepalLengthCm",
    "business_interpretation": "Estimating sepal length helps researchers reconstruct the physical profile of a specimen if parts of the flower are damaged or missing in herbarium samples."
  }},
  {{
    "problem_id": 5,
    "problem_name": "Sepal Width Prediction",
    "task_type": "Regression",
    "inputs": [
      "SepalLengthCm",
      "PetalLengthCm",
      "PetalWidthCm",
      "Species"
    ],
    "target": "SepalWidthCm",
    "business_interpretation": "Predicting the width of the sepal provides insights into the structural integrity and symmetry of the flower, which are often markers for specific environmental adaptations."
  }}
"

"""


    
    # 4️⃣ Call Gemini model
        response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
        # print("Gemini response:", response.text)
        # ml_tasks = infer_ml_tasks_from_csv(uploaded_file, GOOGLE_API_KEY)
        data = json.loads(response.text)

        return JsonResponse({'data': data
}, safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
        
        uploaded_file = request.FILES['file']
        if not uploaded_file.name.endswith('.csv'):
            return JsonResponse({'error': 'File must be a CSV'}, status=400)
        
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        try:
            df = pd.read_csv(file_path)
            head_data = profile_dataset(df)
            # os.remove(file_path)
            return JsonResponse({'data': head_data})
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def simple_option(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)

        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name.lower()

        # Load file
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return JsonResponse(
                {'error': 'File must be a CSV or Excel file (.csv, .xls, .xlsx)'},
                status=400
            )

        # Parse problem definition
        try:
            problem_raw = request.POST.get('problem')
            if not problem_raw:
                return JsonResponse({'error': 'No problem definition provided'}, status=400)
            problem = json.loads(problem_raw)
            print("problem", problem)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid problem JSON'}, status=400)

        # Fill NaN values in object (text) columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('')

        # ---------- PREPROCESSING ----------
        system = IntelligentPreprocessingSystem(df, problem)
        prep_result = system.run()

        X_ready = prep_result["preprocessor"].fit_transform(prep_result["X"])
        y_ready = prep_result["y"]

        # ---------- MODELING ----------
        automl = AutoModelingSystem(X_ready, y_ready, problem)
        results = automl.run()

        # ---------- FORMAT FRONTEND DATA ----------
        formatter = FrontendResultFormatter(results, prep_result["X"], prep_result["y"], problem)
        frontend_output = formatter.format()

        # ---------- SAVE BEST MODEL ----------
        best_model = results["BestModel"]["model"]
        print("best model is", best_model)
        # Ensure media directory exists
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        model_path = os.path.join(settings.MEDIA_ROOT, f"best_model_{problem['problem_id']}.pkl")
        joblib.dump(best_model, model_path)

        frontend_output["best_model"]["model_file"] = model_path
        print("model saved at", frontend_output)
        return JsonResponse(frontend_output, safe=False)

    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def preprocess_data(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
        if 'problem' not in request.FILES:
         print("not there")
        problem_raw = request.POST.get('problem')
        print("problem raw", type(problem_raw))
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name.lower()

        # Load file
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return JsonResponse(
                {'error': 'File must be a CSV or Excel file (.csv, .xls, .xlsx)'},
                status=400
            )
        problem = {
    "problem_id": 1,
    "problem_name": "Market Close Price Prediction",
    "task_type": "Regression",
    "inputs": [
      "Date",
      "Index_Name",
      "Country",
      "Open",
      "High",
      "Low",
      "Volume",
      "Daily_Change_Percent"
    ],
    "target": "Close",
    "business_interpretation": "Predicting the closing price of a stock market index is essential for traders and portfolio managers to determine the daily settlement value and evaluate the performance of their investments at the end of the trading session."
  }
        problem = json.loads(problem_raw)
        # Fill NaN values in object (text) columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('')

        # ---------- PREPROCESSING ----------
        system = IntelligentPreprocessingSystem(df, problem)
        result = system.run()

        df_clean = DataCleaner(df, problem).clean()
        X = result["X"]
        y = result["y"]

# Generate dashboard data
        graph1 = get_target_distribution_data(df_clean, y, problem)
        graph2 = get_feature_target_relationship_data(X, y, result["column_types"], problem)
        graph3 = get_preprocessing_impact_data(df, df_clean, result["column_types"], result["strategies"])
        summary_box = get_readiness_summary(result["validation_report"], result["column_types"], result["strategies"])

        dashboard_payload = {
    "target_distribution": graph1,
    "feature_relationships": graph2,
    "preprocessing_impact": graph3,
    "readiness_summary": summary_box
}
        return JsonResponse({"preprocessed data":dashboard_payload}, safe=False)

    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)