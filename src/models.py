"""
Machine Learning Model Implementations

This module provides wrapper functions for training and evaluating
ML models using PySpark MLlib.
"""

from pyspark.ml.classification import NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer, StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd


def create_spark_session(app_name="HotelBookingML"):
    """
    Create and return Spark session.
    
    Args:
        app_name (str): Application name
    
    Returns:
        SparkSession: Spark session object
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    return spark


def prepare_spark_dataframe(spark, df, target_col='is_canceled'):
    """
    Prepare pandas DataFrame for Spark ML.
    
    Args:
        spark (SparkSession): Spark session
        df (pd.DataFrame): Pandas DataFrame
        target_col (str): Target column name
    
    Returns:
        pyspark.sql.DataFrame: Spark DataFrame
    """
    # Convert pandas to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    
    # Ensure target column is integer
    spark_df = spark_df.withColumn(target_col, col(target_col).cast("integer"))
    
    return spark_df


def train_naive_bayes(train_df, features_col='features', label_col='is_canceled'):
    """
    Train Naive Bayes model.
    
    Args:
        train_df (pyspark.sql.DataFrame): Training data
        features_col (str): Name of features column
        label_col (str): Name of label column
    
    Returns:
        NaiveBayesModel: Trained model
    """
    nb = NaiveBayes(
        featuresCol=features_col,
        labelCol=label_col
    )
    
    model = nb.fit(train_df)
    return model


def train_decision_tree(train_df, features_col='features', label_col='is_canceled', max_depth=10):
    """
    Train Decision Tree model.
    
    Args:
        train_df (pyspark.sql.DataFrame): Training data
        features_col (str): Name of features column
        label_col (str): Name of label column
        max_depth (int): Maximum depth of tree
    
    Returns:
        DecisionTreeClassificationModel: Trained model
    """
    dt = DecisionTreeClassifier(
        featuresCol=features_col,
        labelCol=label_col,
        maxDepth=max_depth,
        impurity='gini'
    )
    
    model = dt.fit(train_df)
    return model


def evaluate_model(predictions, label_col='is_canceled', prediction_col='prediction'):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        predictions (pyspark.sql.DataFrame): Predictions DataFrame
        label_col (str): Name of label column
        prediction_col (str): Name of prediction column
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Binary classification evaluator for AUC
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol='rawPrediction',
        metricName='areaUnderROC'
    )
    
    # Multiclass evaluator for other metrics
    multiclass_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='accuracy'
    )
    
    metrics = {
        'accuracy': multiclass_evaluator.evaluate(predictions),
        'auc': binary_evaluator.evaluate(predictions)
    }
    
    # Calculate precision, recall, F1
    for metric_name in ['weightedPrecision', 'weightedRecall', 'f1']:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName=metric_name
        )
        metrics[metric_name] = evaluator.evaluate(predictions)
    
    return metrics


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained model (Decision Tree or Random Forest)
        feature_names (list): List of feature names
    
    Returns:
        pd.DataFrame: DataFrame with feature importance
    """
    try:
        importances = model.featureImportances.toArray()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return importance_df
    except AttributeError:
        print("Model does not support feature importance extraction")
        return None

