import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import (
    confusion_matrix,
)
from numpy import ndarray
from scipy.stats import pearsonr, pointbiserialr, spearmanr
from sklearn.feature_selection import mutual_info_classif
from optuna.integration import XGBoostPruningCallback
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Optional, Any
from imblearn.pipeline import Pipeline as ImbPipeline
import re


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces memory usage by converting data-types"""
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df


def compute_correlations_and_pvalues(
    df: pd.DataFrame, target: pd.Series, method="pearson"
) -> pd.DataFrame:
    """
    Computes correlation of features vs target variable.
    Returns correlation values as well as p-values.
    """
    results = []
    clean_df = df.fillna(df.median())
    for feature in clean_df.columns:
        if method == "pearson":
            corr, p_value = pearsonr(clean_df[feature], clean_df[target])
        elif method == "pointbiserial":
            corr, p_value = pointbiserialr(clean_df[feature], clean_df[target])
        elif method == "spearman":
            corr, p_value = spearmanr(clean_df[feature], clean_df[target])

        else:
            raise ValueError(
                "Method must be either 'pearson','pointbiserial', or'spearman'."
            )
        results.append((feature, corr, p_value))
    results_df = pd.DataFrame(
        results, columns=["feature", "correlation", "p_value"]
    ).sort_values(by="correlation", ascending=False)

    return results_df


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes count and percentage of
    null values for given DataFrame
    """
    num_null_values = df.isnull().sum()
    pct_null_values = (df.isnull().sum() / df.shape[0]) * 100
    return pd.DataFrame(
        {
            "feature": df.columns,
            "null_values_count": num_null_values.values,
            "null_values_pct": pct_null_values.values,
        }
    ).sort_values(by="null_values_pct", ascending=False)


def get_columns_starting_with(df: pd.DataFrame, keyword: str) -> list[str]:
    """Extracts column names starting with a string"""
    return [col for col in df.columns if col.startswith(keyword)]


def get_numerical_categorical_columns(
    df: pd.DataFrame, binary_as_categorical: bool = False
) -> list[str]:
    """
    Extracts numerical and categorical columns
    with the option to treat binary as categorical
    """
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if binary_as_categorical:
        binary_cols = [
            col for col in numerical_cols if set(df[col].dropna().unique()) == {0, 1}
        ]
        numerical_cols = [col for col in numerical_cols if col not in binary_cols]
        categorical_cols += binary_cols
    return numerical_cols, categorical_cols


def get_ft_importances(X_train: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Gets feature importance DataFrame for given model"""
    importances_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    return importances_df


def multiple_test_chi2(X: pd.DataFrame, y: pd.Series) -> dict[str, list[float]]:
    """Performs chi-square tests for a list of features against a target."""
    results = {}
    for feature in X.columns:
        contingency_table = pd.crosstab(X[feature], y)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results[feature] = [p_value, chi2]
    return results


def clean_days_columns(column: pd.Series) -> pd.Series:
    """
    Replaces outlier values
    for the days since column with nan.
    """
    return column.mask(column > 365000, other=np.nan)


def replace_unknown_values(column: pd.Series) -> pd.Series:
    """Replaces unknown values with np.nan for any column type."""
    column = column.replace(["XNA", "Unknown", "XAP"], np.nan)
    if pd.api.types.is_categorical_dtype(column):
        column = column.cat.remove_unused_categories()

    return column


def bin_ages(ages: pd.Series) -> pd.Series:
    """
    Bins a Series of ages into their respective groups
    and returns the new binned ages as a Series.
    """
    bins = [20, 25, 35, 45, 55, np.inf]
    labels = ["20-25", "26-35", "36-45", "46-55", "56+"]
    age_binned = pd.cut(ages, bins=bins, labels=labels)
    return age_binned


def convert_days_to_years(column: pd.Series) -> pd.Series:
    """Converts days to years"""
    return abs(column) / 365


def clean_applications_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans applications dataset by applying
    various cleaning and transformation functions.
    """
    df["AGE_YEARS"] = convert_days_to_years(df["DAYS_BIRTH"])
    df["YEARS_EMPLOYED"] = convert_days_to_years(df["DAYS_EMPLOYED"])
    df.drop("DAYS_BIRTH", axis=1, inplace=True)
    df[["CODE_GENDER", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS"]] = df[
        ["CODE_GENDER", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS"]
    ].apply(replace_unknown_values)
    days_columns = get_columns_starting_with(df, "DAYS")
    df[days_columns] = df[days_columns].apply(clean_days_columns)
    house_apt_cols_to_drop = [
        col for col in df.columns if col.endswith(("MEDI", "MODE"))
    ]
    df = housing_education_mapping(df)
    df = df.drop(columns=house_apt_cols_to_drop)
    return df


def clean_previous_apps_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans previous applications dataset"""
    name_cols = list(get_columns_starting_with(df, "NAME")) + ["CODE_REJECT_REASON"]
    days_cols = get_columns_starting_with(df, "DAYS")
    df[name_cols] = df[name_cols].apply(replace_unknown_values)

    df[days_cols] = df[days_cols].apply(clean_days_columns)
    return df


def clean_pos_cash_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces unknown values pos cash dataset"""
    df["NAME_CONTRACT_STATUS"] = replace_unknown_values(df["NAME_CONTRACT_STATUS"])
    return df


def agg_dataframes(
    main_df: pd.DataFrame, df_to_agg: pd.DataFrame, index_col: str, df_name: str
) -> pd.DataFrame:
    """
    Merges `main_df` with aggregated features from `df_to_agg` based on `index_col`.
    Numerical features are aggregated with common statistics;
    categoricals with <= 10 unique values are count-aggregated.
    """
    num_cols, cat_cols = get_numerical_categorical_columns(
        df_to_agg, binary_as_categorical=True
    )
    agg_funcs_num = ["count", "mean", "max", "min", "sum", "std"]
    cols_remove = [
        col
        for col in num_cols
        if col.startswith("SK") and "DPD" not in col and col != index_col
    ]
    num_agg = (
        df_to_agg[num_cols]
        .drop(columns=cols_remove)
        .groupby(index_col)
        .agg(agg_funcs_num)
        .reset_index()
    )

    num_agg.columns = ["SK_ID_CURR"] + [
        f"{df_name}_{col[0]}_{col[1].upper()}" for col in num_agg.columns.ravel()[1:]
    ]

    merged_df = main_df.merge(num_agg, on=index_col, how="left")

    for col in cat_cols:
        if df_to_agg[col].nunique() <= 10:
            cat_agg = (
                df_to_agg.groupby([index_col, col], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            cat_agg.columns = [
                f"{df_name}_COUNT_{category.upper() if isinstance(category, str) else category}"
                for category in cat_agg.columns
            ]
            cat_agg.reset_index(inplace=True)
            cat_agg.columns = [
                re.sub(r"[^a-zA-Z0-9]", "_", col_name) for col_name in cat_agg.columns
            ]
            merged_df = merged_df.merge(cat_agg, on=index_col, how="left")
    merged_df = reduce_memory_usage(merged_df)
    return merged_df


def get_confusion_matrix_df(
    y_test: ndarray, y_preds: ndarray, normalize=None
) -> pd.DataFrame:
    """Creates confusion matrix DataFrame"""
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, y_preds, normalize=normalize),
        columns=["Predicted Negative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
    return conf_matrix


def housing_education_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps education and housing values to shorter strings.
    Allows for plotting of these variables.
    """
    simplified_education_map = {
        "Secondary / secondary special": "Secondary",
        "Higher education": "Higher",
        "Incomplete Higher": "Incomplete",
        "Lower secondary": "Lower Secondary",
        "Academic degree": "Academic",
    }
    simplified_housing_map = {
        "House / apartment": "House/Apt",
        "With parents": "Parents",
        "Municipal apartment": "Municipal",
        "Rented apartment": "Rented",
        "Office apartment": "Office",
        "Co-op apartment": "Co-op",
    }

    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].replace(
        simplified_education_map
    )
    df["NAME_HOUSING_TYPE"] = df["NAME_HOUSING_TYPE"].replace(simplified_housing_map)
    return df


def safe_division(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Returns nan if division by 0.
    This is useful not to have any infinity values in the dataset
    """
    return np.where(denominator != 0, numerator / denominator, np.nan)


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers various features from existing aggregated dataframe"""
    df_copy = df.copy()
    df_copy["EXT_SOURCES_MEAN"] = df_copy[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].mean(axis=1)
    df_copy["EXT_SOURCES_STD"] = df_copy[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].std(axis=1)
    df_copy["CREDIT_TO_INCOME_RATIO"] = safe_division(
        df_copy["AMT_CREDIT"], df_copy["AMT_INCOME_TOTAL"]
    )
    df_copy["ANNUITY_TO_CREDIT"] = safe_division(
        df_copy["AMT_ANNUITY"], df_copy["AMT_CREDIT"]
    )
    df_copy["EMPLOYMENT_TO_AGE_RATIO"] = safe_division(
        abs(df_copy["DAYS_EMPLOYED"]), df_copy["AGE_YEARS"]
    )
    df_copy["CREDIT_TO_AGE_RATIO"] = safe_division(
        df_copy["AMT_CREDIT"], df_copy["AGE_YEARS"]
    )
    df_copy["DEBT_TO_CREDIT_RATIO"] = safe_division(
        df_copy["BUREAU_AMT_CREDIT_SUM_DEBT_MEAN"], df_copy["AMT_CREDIT"]
    )
    return df_copy


def mutual_information_scores(
    X_train: pd.DataFrame, y_train: pd.Series, preprocessor
) -> pd.DataFrame:
    """
    Computes mutual information scores for training data and returns a DataFrame.
    """
    preprocessor.fit(X_train, y_train)
    X_preprocessed = preprocessor.transform(X_train)
    mi_scores = mutual_info_classif(X_preprocessed, y_train)
    mi_scores_df = pd.DataFrame(
        {"feature": preprocessor.get_feature_names_out(), "mi_score": mi_scores}
    )
    mi_scores_df["feature"] = (
        mi_scores_df["feature"]
        .str.replace("categorical|numerical|pass|_", " ", regex=True)
        .str.strip()
    )
    mi_scores_df = mi_scores_df.sort_values(by="mi_score", ascending=False).reset_index(
        drop=True
    )
    return mi_scores_df


def get_objective_xgb(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> callable:
    def objective(trial) -> float:
        xgb_param_grid = {
            "n_estimators": 750,
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 10),
            "scale_pos_weight": trial.suggest_float("xgb_scale_pos_weight", 1, 5),
            "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lambda", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("alpha", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "eval_metric": "auc",
            "use_label_encoder": False,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 2,
            "enable_categorical": True,
        }

        model = xgb.XGBClassifier(**xgb_param_grid)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-auc")
        model.set_params(callbacks=[pruning_callback], early_stopping_rounds=30)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        y_probas = model.predict_proba(X_val)[:, -1]
        score = roc_auc_score(y_val, y_probas)
        return score

    return objective


def create_numerical_pipeline(imputer: TransformerMixin) -> Pipeline:
    """Creates a pipeline for processing numerical data."""
    return Pipeline([("imputer", imputer)])


def create_categorical_pipeline(
    imputer: TransformerMixin, encoder: TransformerMixin
) -> Pipeline:
    """Creates a pipeline for processing categorical data."""
    return Pipeline([("imputer", imputer), ("encoder", encoder)])


def create_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str],
    imputer_numerical: TransformerMixin = SimpleImputer(strategy="median"),
    imputer_categorical: TransformerMixin = SimpleImputer(strategy="most_frequent"),
    encoder: TransformerMixin = TargetEncoder(),
    scaler: Optional[TransformerMixin] = None,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for preprocessing numerical and categorical columns.
    Allows customization of imputation, encoding, and optional scaling. Returns a
    ColumnTransformer if no scaler is provided, otherwise a Pipeline with scaling applied.
    """

    transformers = [
        (
            "numerical",
            create_numerical_pipeline(imputer_numerical),
            numerical_cols,
        ),
        (
            "categorical",
            create_categorical_pipeline(imputer_categorical, encoder),
            categorical_cols,
        ),
    ]
    if scaler:
        return Pipeline(
            [
                (
                    "preprocessing",
                    ColumnTransformer(transformers, remainder="passthrough"),
                ),
                ("scaler", scaler),
            ]
        )

    return ColumnTransformer(transformers, remainder="passthrough")


def filter_columns_by_null_threshold(
    df: pd.DataFrame, threshold: int = 40
) -> pd.DataFrame:
    """Filter columns in a DataFrame based on a null percentage threshold."""
    null_percentage = (df.isnull().sum() / df.shape[0]) * 100
    return df.loc[:, null_percentage < threshold]


def create_preprocessor_with_null_threshold(
    X_train: pd.DataFrame, threshold: int = 40
) -> ColumnTransformer:
    """Build preprocessors for datasets based on null value threshold filtering."""
    filtered_df = filter_columns_by_null_threshold(X_train, threshold)
    numerical_cols, categorical_cols = get_numerical_categorical_columns(filtered_df)

    return create_preprocessor(
        numerical_cols=numerical_cols, categorical_cols=categorical_cols
    )


def get_highly_correlated_features(df, threshold=0.8):
    """Returns pairs of highly correlated features based on a threshold"""
    corr_matrix = df.corr(numeric_only=True).abs()
    correlated_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                correlated_pairs.append(
                    (
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    correlated_features_df = pd.DataFrame(
        correlated_pairs, columns=["Feature_1", "Feature_2", "Correlation"]
    )
    return correlated_features_df


def build_pipeline(
    preprocessor: ColumnTransformer, model: Any, sampling="passthrough"
) -> Pipeline:
    """
    Builds a pipeline with the option of using a
    sampling technique (e.g, SMOTE,UnderSampling, etc.)
    """
    if sampling:
        return ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("sampling", sampling),
                ("classifier", model),
            ]
        )
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def get_preds_and_probas(
    model: Any, X_val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Makes predictions and returns the predicted probabilities and labels"""
    y_preds = model.predict(X_val)
    y_probas = model.predict_proba(X_val)[:, -1]
    return y_preds, y_probas


def mutiple_test_mann_whitney(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Performs Mann-Whitney U tests for a list of features against a target."""
    results = {}
    for feature in X.columns:
        non_null_indices = X[feature].notna() & y.notna()
        feature_data = X.loc[non_null_indices, feature]
        target_data = y[non_null_indices]
        group1 = feature_data[target_data == 0]
        group2 = feature_data[target_data == 1]
        u_stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
        results[feature] = [p_value, u_stat]
    return results


def calculate_overlap(
    df1: pd.DataFrame, df2: pd.DataFrame, key: str = "SK_ID_CURR"
) -> int:
    """Calculate the overlap between two DataFrames based on a key."""
    overlap = len(set(df1[key]) & set(df2[key]))
    return overlap
