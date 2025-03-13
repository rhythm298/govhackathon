# Disease Outbreak Prediction System
# Main Components: Data Pipeline, ML Models, and Alert System

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import xgboost as xgb
from prophet import Prophet
import geopandas as gpd
from pymongo import MongoClient
import redis
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import boto3
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import json
import logging
import os
import datetime
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from twilio.rest import Client
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("disease_prediction_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("disease_outbreak_prediction")

# Initialize Flask app for the API
app = Flask(__name__)
CORS(app)

# Database Configuration
class DatabaseManager:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGODB_URI"))
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            password=os.getenv("REDIS_PASSWORD")
        )
        self.es_client = Elasticsearch([os.getenv("ELASTICSEARCH_URI")])
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        
        # Initialize Spark session for big data processing
        self.spark = SparkSession.builder \
            .appName("DiseaseOutbreakPrediction") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
            .getOrCreate()
        
        logger.info("Database connections initialized successfully")
    
    def fetch_hospital_data(self, start_date, end_date):
        """Fetch hospital admission data from MongoDB"""
        db = self.mongo_client["healthcare_data"]
        collection = db["hospital_admissions"]
        
        query = {
            "admission_date": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        
        cursor = collection.find(query)
        data = list(cursor)
        df = pd.DataFrame(data)
        
        logger.info(f"Fetched {len(df)} hospital records between {start_date} and {end_date}")
        return df
    
    def fetch_weather_data(self, region_id, start_date, end_date):
        """Fetch weather data from Redis cache or external API"""
        cache_key = f"weather:{region_id}:{start_date}:{end_date}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            logger.info(f"Using cached weather data for {region_id}")
            return pd.read_json(cached_data)
        
        # If not in cache, fetch from external API (placeholder)
        weather_api_url = os.getenv("WEATHER_API_URL")
        response = requests.get(
            f"{weather_api_url}/historical",
            params={
                "region_id": region_id,
                "start_date": start_date,
                "end_date": end_date,
                "api_key": os.getenv("WEATHER_API_KEY")
            }
        )
        
        if response.status_code == 200:
            weather_data = response.json()
            df = pd.DataFrame(weather_data["data"])
            
            # Cache the result for 6 hours
            self.redis_client.setex(
                cache_key,
                21600,  # 6 hours in seconds
                df.to_json()
            )
            
            logger.info(f"Fetched and cached weather data for {region_id}")
            return df
        else:
            logger.error(f"Failed to fetch weather data: {response.status_code}")
            return pd.DataFrame()
    
    def store_prediction_results(self, results):
        """Store prediction results in Elasticsearch for quick retrieval and visualization"""
        for result in results:
            self.es_client.index(
                index="disease_predictions",
                document=result
            )
        
        logger.info(f"Stored {len(results)} prediction results in Elasticsearch")
    
    def upload_model_to_s3(self, model_path, model_name):
        """Upload trained model to S3 bucket for deployment"""
        bucket_name = os.getenv("S3_MODEL_BUCKET")
        s3_key = f"models/{model_name}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.h5"
        
        self.s3_client.upload_file(
            model_path,
            bucket_name,
            s3_key
        )
        
        logger.info(f"Uploaded model {model_name} to S3 bucket {bucket_name}")
        return s3_key

# Data Processing Pipeline
class DataPipeline:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    def fetch_and_merge_data(self, region_ids, start_date, end_date):
        """Fetch data from multiple sources and merge them for analysis"""
        # Get hospital admission data
        hospital_data = self.db_manager.fetch_hospital_data(start_date, end_date)
        
        # Get weather data for each region
        weather_frames = []
        for region_id in region_ids:
            weather_df = self.db_manager.fetch_weather_data(region_id, start_date, end_date)
            weather_df['region_id'] = region_id
            weather_frames.append(weather_df)
        
        weather_data = pd.concat(weather_frames)
        
        # Fetch population movement data (placeholder)
        pop_movement_api_url = os.getenv("POPULATION_MOVEMENT_API")
        response = requests.get(
            pop_movement_api_url,
            params={
                "regions": ','.join(region_ids),
                "start_date": start_date,
                "end_date": end_date,
                "api_key": os.getenv("POP_MOVEMENT_API_KEY")
            }
        )
        
        if response.status_code == 200:
            movement_data = pd.DataFrame(response.json()["data"])
        else:
            logger.warning("Failed to fetch population movement data")
            movement_data = pd.DataFrame()
        
        # Fetch social media trends data (placeholder)
        social_media_api_url = os.getenv("SOCIAL_MEDIA_API")
        response = requests.get(
            social_media_api_url,
            params={
                "keywords": "fever,cough,illness,outbreak,disease",
                "regions": ','.join(region_ids),
                "start_date": start_date,
                "end_date": end_date,
                "api_key": os.getenv("SOCIAL_MEDIA_API_KEY")
            }
        )
        
        if response.status_code == 200:
            social_data = pd.DataFrame(response.json()["data"])
        else:
            logger.warning("Failed to fetch social media trends data")
            social_data = pd.DataFrame()
        
        # Load geographic data for spatial analysis
        gis_data = gpd.read_file(os.getenv("GIS_SHAPEFILE_PATH"))
        
        # Merge all datasets
        # This is a simplified example - real implementation would handle data alignment
        # across different time periods and spatial resolutions
        merged_data = self._merge_datasets(
            hospital_data, 
            weather_data, 
            movement_data, 
            social_data, 
            gis_data
        )
        
        logger.info(f"Created merged dataset with {len(merged_data)} records")
        return merged_data
    
    def _merge_datasets(self, hospital_df, weather_df, movement_df, social_df, gis_df):
        """Perform the actual merging of different datasets"""
        # Convert dates to common format
        for df in [hospital_df, weather_df, movement_df, social_df]:
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # First merge hospital data with weather based on region and date
        if not hospital_df.empty and not weather_df.empty:
            merged = pd.merge(
                hospital_df,
                weather_df,
                on=['region_id', 'date'],
                how='left'
            )
        else:
            merged = hospital_df.copy() if not hospital_df.empty else pd.DataFrame()
        
        # Add population movement data
        if not movement_df.empty:
            merged = pd.merge(
                merged,
                movement_df,
                on=['region_id', 'date'],
                how='left'
            )
        
        # Add social media trends
        if not social_df.empty:
            merged = pd.merge(
                merged,
                social_df,
                on=['region_id', 'date'],
                how='left'
            )
        
        # Add geographic features
        if not gis_df.empty and not merged.empty and 'region_id' in merged.columns:
            # Convert GIS data to have the same region_id format
            gis_df = gis_df.rename(columns={'ID': 'region_id'})
            
            # Extract relevant geographic features
            geo_features = gis_df[['region_id', 'area', 'population_density', 'elevation']]
            
            merged = pd.merge(
                merged,
                geo_features,
                on='region_id',
                how='left'
            )
        
        # Handle missing values
        merged.fillna({
            'temperature': merged['temperature'].mean() if 'temperature' in merged.columns else 0,
            'humidity': merged['humidity'].mean() if 'humidity' in merged.columns else 0,
            'population_density': merged['population_density'].mean() if 'population_density' in merged.columns else 0,
            'social_mentions': 0  # Default for social media mentions
        }, inplace=True)
        
        return merged
    
    def preprocess_data(self, df):
        """Preprocess data for modeling"""
        # Handle categorical variables
        categorical_cols = ['disease_type', 'region_id', 'symptom_category']
        numeric_cols = ['temperature', 'humidity', 'precipitation', 'admission_count', 
                       'population_density', 'social_mentions', 'movement_index']
        
        # Check if all categorical columns exist in the dataframe
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if categorical_cols:
            # Fit and transform categorical variables
            encoded = self.encoder.fit_transform(df[categorical_cols])
            
            # Create dataframe with encoded categorical variables
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(categorical_cols)
            )
            
            # Drop original categorical columns and join encoded ones
            df_processed = df.drop(columns=categorical_cols).reset_index(drop=True)
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
        else:
            df_processed = df.copy()
        
        # Scale numeric features
        if numeric_cols:
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        
        # Create time-based features
        if 'date' in df.columns:
            df_processed['day_of_week'] = df['date'].dt.dayofweek
            df_processed['month'] = df['date'].dt.month
            df_processed['day'] = df['date'].dt.day
            
            # Drop the original date column
            df_processed.drop(columns=['date'], inplace=True)
        
        logger.info(f"Preprocessed data with {df_processed.shape[1]} features")
        return df_processed
    
    def create_time_series_features(self, df, target_col, window_size=7):
        """Create features for time series forecasting"""
        # Group by region and date
        if 'region_id' in df.columns and target_col in df.columns:
            # Sort by date
            df = df.sort_values('date')
            
            # Create lagged features
            for lag in range(1, window_size + 1):
                df[f'{target_col}_lag_{lag}'] = df.groupby('region_id')[target_col].shift(lag)
            
            # Create rolling statistics
            df[f'{target_col}_rolling_mean'] = df.groupby('region_id')[target_col].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
            
            df[f'{target_col}_rolling_std'] = df.groupby('region_id')[target_col].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).std()
            )
            
            # Drop rows with NaN values (from the lagging operation)
            df.dropna(inplace=True)
        
        return df
    
    def prepare_spark_dataframe(self, df):
        """Convert pandas DataFrame to Spark DataFrame for big data processing"""
        spark_df = self.db_manager.spark.createDataFrame(df)
        logger.info(f"Created Spark DataFrame with {spark_df.count()} rows")
        return spark_df

# Machine Learning Models
class ModelManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.models = {}
        self.models_metadata = {}
    
    def build_lstm_model(self, input_shape, output_units=1):
        """Build an LSTM model for time series forecasting"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built LSTM model with input shape {input_shape}")
        return model
    
    def build_cnn_lstm_model(self, input_shape, output_units=1):
        """Build a CNN-LSTM hybrid model for sequence prediction with spatial features"""
        # Input layer
        input_layer = Input(shape=input_shape)
        
        # CNN layers for feature extraction
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        
        # LSTM layers for sequence learning
        lstm1 = LSTM(32, return_sequences=True)(pool2)
        drop1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(16)(drop1)
        drop2 = Dropout(0.2)(lstm2)
        
        # Dense layers for prediction
        dense1 = Dense(16, activation='relu')(drop2)
        output_layer = Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')(dense1)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built CNN-LSTM hybrid model with input shape {input_shape}")
        return model
    
    def build_xgboost_model(self):
        """Build an XGBoost model for classification/regression"""
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        logger.info("Built XGBoost classifier model")
        return model
    
    def build_spark_rf_model(self, feature_columns, label_column):
        """Build a Spark RandomForest model for big data processing"""
        # Assemble features into a single vector column
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )
        
        # Create and configure the model
        rf = SparkRandomForestClassifier(
            labelCol=label_column,
            featuresCol="features",
            numTrees=100,
            maxDepth=5,
            seed=42
        )
        
        logger.info(f"Built Spark RandomForest model with {len(feature_columns)} features")
        return [assembler, rf]  # Return as a pipeline list
    
    def build_prophet_model(self):
        """Build a Prophet model for time series forecasting"""
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        logger.info("Built Prophet forecasting model")
        return model
    
    def train_model(self, model_type, X_train, y_train, **kwargs):
        """Train a model based on the specified type"""
        if model_type == "lstm":
            # Reshape data for LSTM [samples, time steps, features]
            if len(X_train.shape) < 3:
                time_steps = kwargs.get('time_steps', 7)
                n_features = X_train.shape[1]
                X_train_reshaped = X_train.values.reshape(-1, time_steps, n_features // time_steps)
            else:
                X_train_reshaped = X_train
            
            # Get or build LSTM model
            input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
            model = self.build_lstm_model(input_shape, output_units=1 if len(y_train.shape) == 1 else y_train.shape[1])
            
            # Training configuration
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            checkpoint = ModelCheckpoint(
                f"models/lstm_model_{int(time.time())}.h5",
                save_best_only=True,
                monitor='val_loss'
            )
            
            # Train model
            history = model.fit(
                X_train_reshaped, y_train,
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32),
                validation_split=0.2,
                callbacks=[early_stopping, checkpoint]
            )
            
            # Save model and metadata
            model_path = f"models/lstm_model_{int(time.time())}.h5"
            model.save(model_path)
            
            # Upload to S3
            s3_key = self.db_manager.upload_model_to_s3(model_path, "lstm_disease_prediction")
            
            self.models["lstm"] = model
            self.models_metadata["lstm"] = {
                "training_date": datetime.datetime.now().isoformat(),
                "input_shape": input_shape,
                "performance": {
                    "val_loss": min(history.history['val_loss']),
                    "val_accuracy": max(history.history['val_accuracy'])
                },
                "s3_path": s3_key
            }
            
            logger.info(f"Trained LSTM model with validation accuracy: {max(history.history['val_accuracy']):.4f}")
            
        elif model_type == "xgboost":
            model = self.build_xgboost_model()
            model.fit(X_train, y_train)
            
            # Save model
            model_path = f"models/xgboost_model_{int(time.time())}.json"
            model.save_model(model_path)
            
            # Upload to S3
            s3_key = self.db_manager.upload_model_to_s3(model_path, "xgboost_disease_prediction")
            
            self.models["xgboost"] = model
            self.models_metadata["xgboost"] = {
                "training_date": datetime.datetime.now().isoformat(),
                "parameters": model.get_params(),
                "s3_path": s3_key
            }
            
            logger.info("Trained XGBoost model")
            
        elif model_type == "prophet":
            # Prepare data for Prophet (needs 'ds' and 'y' columns)
            prophet_data = pd.DataFrame({
                'ds': kwargs.get('dates'),
                'y': y_train
            })
            
            model = self.build_prophet_model()
            
            # Add regressors if provided
            if 'regressors' in kwargs and kwargs['regressors'] is not None:
                for regressor_name, regressor_values in kwargs['regressors'].items():
                    prophet_data[regressor_name] = regressor_values
                    model.add_regressor(regressor_name)
            
            model.fit(prophet_data)
            
            self.models["prophet"] = model
            self.models_metadata["prophet"] = {
                "training_date": datetime.datetime.now().isoformat(),
                "parameters": {
                    "changepoint_prior_scale": model.changepoint_prior_scale,
                    "seasonality_mode": model.seasonality_mode
                }
            }
            
            logger.info("Trained Prophet model")
            
        elif model_type == "spark_rf":
            # Get features and label columns
            feature_columns = kwargs.get('feature_columns')
            label_column = kwargs.get('label_column')
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.db_manager.spark.createDataFrame(
                pd.concat([X_train, y_train.rename(label_column)], axis=1)
            )
            
            # Build and train Spark model
            assembler, rf = self.build_spark_rf_model(feature_columns, label_column)
            
            # Create pipeline and fit
            from pyspark.ml import Pipeline
            pipeline = Pipeline(stages=[assembler, rf])
            model = pipeline.fit(spark_df)
            
            # Save model
            model_path = f"models/spark_rf_model_{int(time.time())}"
            model.write().overwrite().save(model_path)
            
            self.models["spark_rf"] = model
            self.models_metadata["spark_rf"] = {
                "training_date": datetime.datetime.now().isoformat(),
                "parameters": {
                    "numTrees": rf.getNumTrees(),
                    "maxDepth": rf.getMaxDepth()
                },
                "model_path": model_path
            }
            
            logger.info("Trained Spark RandomForest model")
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def evaluate_model(self, model_type, model, X_test, y_test, **kwargs):
        """Evaluate a trained model"""
        results = {}
        
        if model_type == "lstm":
            # Reshape data for LSTM if needed
            if len(X_test.shape) < 3:
                time_steps = kwargs.get('time_steps', 7)
                n_features = X_test.shape[1]
                X_test_reshaped = X_test.values.reshape(-1, time_steps, n_features // time_steps)
            else:
                X_test_reshaped = X_test
            
            # Make predictions
            y_pred_proba = model.predict(X_test_reshaped)
            
            # Convert probabilities to class labels for classification
            if len(y_test.shape) == 1 or y_test.shape[1] == 1:
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                results = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_pred_proba)
                }
            else:
                # Multi-class metrics
                y_pred = np.argmax(y_pred_proba, axis=1)
                y_test_labels = np.argmax(y_test, axis=1)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                results = {
                    "accuracy": accuracy_score(y_test_labels, y_pred),
                    "precision": precision_score(y_test_labels, y_pred, average='weighted'),
                    "recall": recall_score(y_test_labels, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test_labels, y_pred, average='weighted')
                }
        
        elif model_type == "xgboost":
            # Make predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            results = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }
            
            # Feature importance
            feature_importance = model.feature_importances_
            feature_names = X_test.columns
            
            results["feature_importance"] = {
                name: importance for name, importance in zip(feature_names, feature_importance)
            }
        
        elif model_type == "prophet":
            # Create a future dataframe for prediction
            future = model.make_future_dataframe(periods=len(y_test))
            
            # Add regressor values if any
            if 'regressors' in kwargs and kwargs['regressors'] is not None:
                for regressor_name, regressor_values in kwargs['regressors'].items():
                    future[regressor_name] = regressor_values
            
            # Generate predictions
            forecast = model.predict(future)
            
            # Extract the forecasted values corresponding to the test period
            y_pred = forecast['yhat'].iloc[-len(y_test):].values
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            results = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred)
            }
        
        elif model_type == "spark_rf":
            # Prepare test data
            label_column = kwargs.get('label_column')
            
            # Convert pandas DataFrame to Spark DataFrame
            test_spark_df = self.db_manager.spark.createDataFrame(
                pd.concat([X_test, y_test.rename(label_column)], axis=1)
            )
            
            # Generate predictions
            predictions = model.transform(test_spark_df)
            
            # Evaluate model
            evaluator = MulticlassClassificationEvaluator(
                labelCol=label_column,
                predictionCol="prediction",
                metricName="accuracy"
            )
            accuracy = evaluator.evaluate(predictions)
            
            # Calculate additional metrics
            from pyspark.ml.evaluation import BinaryClassificationEvaluator
            binary_evaluator = BinaryClassificationEvaluator(
                labelCol=label_column,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            roc_auc = binary_evaluator.evaluate(predictions)
            
            results = {
                "accuracy": accuracy,
                "roc_auc": roc_auc
            }
        
        # Log evaluation results
        logger.info(f"Model evaluation results for {model_type}: {results}")
        
        # Update model metadata with evaluation results
        if model_type in self.models_metadata:
            self.models_metadata[model_type]["evaluation"] = results
        
        return results
    
    def predict_outbreak(self, model_type, data, **kwargs):
        """Make outbreak predictions using the trained model"""
        if model_type not in self.models:
            logger.error(f"Model {model_type} not found or not trained")
            return None
        
        model = self.models[model_type]
        
if model_type not in self.models:
            logger.error(f"Model {model_type} not found or not trained")
            return None
        
        model = self.models[model_type]
        
        if model_type == "lstm":
            # Reshape data for LSTM if needed
            if len(data.shape) < 3:
                time_steps = kwargs.get('time_steps', 7)
                n_features = data.shape[1]
                data_reshaped = data.values.reshape(-1, time_steps, n_features // time_steps)
            else:
                data_reshaped = data
            
            # Make predictions
            predictions = model.predict(data_reshaped)
            
            # Process predictions based on output shape
            if predictions.shape[1] == 1:
                # Binary prediction (outbreak probability)
                outbreak_prob = predictions.flatten()
                outbreak_pred = (outbreak_prob > kwargs.get('threshold', 0.5)).astype(int)
                
                results = {
                    "outbreak_probability": outbreak_prob.tolist(),
                    "outbreak_prediction": outbreak_pred.tolist(),
                    "threshold": kwargs.get('threshold', 0.5)
                }
            else:
                # Multi-class prediction (disease types)
                disease_probs = predictions
                disease_pred = np.argmax(disease_probs, axis=1)
                
                results = {
                    "disease_probabilities": disease_probs.tolist(),
                    "disease_prediction": disease_pred.tolist(),
                    "disease_mapping": kwargs.get('disease_mapping', {})
                }
        
        elif model_type == "xgboost":
            # Make predictions
            outbreak_prob = model.predict_proba(data)[:, 1]
            outbreak_pred = model.predict(data)
            
            results = {
                "outbreak_probability": outbreak_prob.tolist(),
                "outbreak_prediction": outbreak_pred.tolist(),
                "threshold": 0.5
            }
        
        elif model_type == "prophet":
            # Prepare future dataframe
            future_periods = kwargs.get('future_periods', 30)
            future = model.make_future_dataframe(periods=future_periods)
            
            # Add regressor values if any
            if 'regressors' in kwargs and kwargs['regressors'] is not None:
                for regressor_name, regressor_values in kwargs['regressors'].items():
                    future[regressor_name] = regressor_values
            
            # Generate predictions
            forecast = model.predict(future)
            
            # Extract forecast components
            results = {
                "forecast_dates": forecast['ds'].astype(str).tolist(),
                "predicted_values": forecast['yhat'].tolist(),
                "lower_bound": forecast['yhat_lower'].tolist(),
                "upper_bound": forecast['yhat_upper'].tolist(),
                "trend": forecast['trend'].tolist()
            }
        
        elif model_type == "spark_rf":
            # Convert data to Spark DataFrame
            feature_columns = kwargs.get('feature_columns')
            spark_df = self.db_manager.spark.createDataFrame(data)
            
            # Generate predictions
            predictions = model.transform(spark_df)
            
            # Convert predictions to pandas for easier processing
            pred_df = predictions.select("prediction", "probability").toPandas()
            
            results = {
                "predictions": pred_df["prediction"].tolist(),
                "probabilities": [p[1] for p in pred_df["probability"]]
            }
        
        # Add metadata to results
        results["model_type"] = model_type
        results["prediction_date"] = datetime.datetime.now().isoformat()
        results["model_metadata"] = self.models_metadata.get(model_type, {})
        
        logger.info(f"Generated outbreak predictions using {model_type} model")
        return results
    
    def interpret_predictions(self, predictions, regions, dates, **kwargs):
        """Interpret prediction results and generate human-readable insights"""
        insights = []
        
        # Extract prediction details
        model_type = predictions.get("model_type")
        
        if model_type in ["lstm", "xgboost"]:
            outbreak_probs = predictions.get("outbreak_probability", [])
            outbreak_preds = predictions.get("outbreak_prediction", [])
            threshold = predictions.get("threshold", 0.5)
            
            # Analyze predictions by region
            for i, (region, prob, pred) in enumerate(zip(regions, outbreak_probs, outbreak_preds)):
                if pred == 1:
                    risk_level = "High" if prob > 0.8 else "Medium"
                    insight = {
                        "region": region,
                        "risk_level": risk_level,
                        "outbreak_probability": prob,
                        "recommendation": self._generate_recommendation(risk_level, region)
                    }
                    insights.append(insight)
        
        elif model_type == "prophet":
            forecast_dates = predictions.get("forecast_dates", [])
            predicted_values = predictions.get("predicted_values", [])
            upper_bound = predictions.get("upper_bound", [])
            
            # Identify potential outbreak periods
            outbreak_threshold = kwargs.get('outbreak_threshold', 1.5)
            baseline = np.mean(predicted_values[:7])  # Use first week as baseline
            
            for i, (date, value, upper) in enumerate(zip(forecast_dates, predicted_values, upper_bound)):
                if value > baseline * outbreak_threshold or upper > baseline * outbreak_threshold * 1.5:
                    risk_level = "High" if value > baseline * 2 else "Medium"
                    
                    insight = {
                        "date": date,
                        "predicted_value": value,
                        "baseline": baseline,
                        "risk_level": risk_level,
                        "recommendation": self._generate_recommendation(risk_level, date=date)
                    }
                    insights.append(insight)
        
        logger.info(f"Generated {len(insights)} insights from prediction results")
        return insights
    
    def _generate_recommendation(self, risk_level, region=None, date=None):
        """Generate recommendations based on risk level and context"""
        recommendations = {
            "High": [
                "Activate emergency response protocols immediately",
                "Deploy rapid response teams to affected area",
                "Increase hospital capacity and prepare isolation facilities",
                "Implement enhanced surveillance and testing",
                "Consider public health restrictions and advisories"
            ],
            "Medium": [
                "Increase surveillance in affected areas",
                "Alert healthcare facilities to prepare for potential cases",
                "Conduct targeted testing in high-risk communities",
                "Review emergency response protocols",
                "Monitor situation closely for changes"
            ],
            "Low": [
                "Maintain routine surveillance activities",
                "Continue regular public health messaging",
                "Review preparedness plans",
                "Monitor for any unusual health patterns"
            ]
        }
        
        selected_recommendations = recommendations.get(risk_level, recommendations["Low"])
        
        # Add context-specific recommendations
        context_specific = []
        if region:
            context_specific.append(f"Focus resources in {region} and surrounding areas")
        
        if date:
            context_specific.append(f"Prepare for potential increase in cases by {date}")
        
        return selected_recommendations + context_specific

# Alert and Notification System
class AlertSystem:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.twilio_client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        self.alert_history = []
    
    def generate_alert(self, insights, region_data):
        """Generate alerts based on prediction insights"""
        alerts = []
        
        for insight in insights:
            risk_level = insight.get("risk_level")
            region = insight.get("region")
            
            if risk_level in ["High", "Medium"]:
                # Get region contacts from region_data
                contacts = self._get_region_contacts(region, region_data)
                
                alert = {
                    "id": f"ALT-{int(time.time())}-{len(self.alert_history) + 1}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "risk_level": risk_level,
                    "region": region,
                    "message": self._generate_alert_message(insight),
                    "recipients": contacts,
                    "status": "pending"
                }
                
                alerts.append(alert)
                self.alert_history.append(alert)
        
        logger.info(f"Generated {len(alerts)} alerts based on prediction insights")
        return alerts
    
    def _get_region_contacts(self, region, region_data):
        """Get contact information for a specific region"""
        if region in region_data:
            return region_data[region].get("contacts", [])
        return []
    
    def _generate_alert_message(self, insight):
        """Generate alert message based on insight data"""
        risk_level = insight.get("risk_level")
        region = insight.get("region", "")
        probability = insight.get("outbreak_probability", 0)
        date = insight.get("date", "")
        
        if risk_level == "High":
            message = f"URGENT: High risk of disease outbreak detected in {region}. "
            message += f"Outbreak probability: {probability:.2f}. "
            if date:
                message += f"Expected outbreak date: {date}. "
            message += "Immediate public health response recommended."
        
        elif risk_level == "Medium":
            message = f"ALERT: Medium risk of disease outbreak detected in {region}. "
            message += f"Outbreak probability: {probability:.2f}. "
            if date:
                message += f"Monitoring period: {date}. "
            message += "Enhanced surveillance and preparedness measures recommended."
        
        else:
            message = f"NOTICE: Low risk of disease activity detected in {region}. "
            message += "Continued monitoring advised."
        
        return message
    
    def send_alerts(self, alerts):
        """Send alerts through multiple channels (SMS, email, dashboard)"""
        for alert in alerts:
            # Update alert status
            alert["status"] = "sending"
            
            # Send SMS alerts
            if os.getenv("ENABLE_SMS_ALERTS") == "true":
                self._send_sms_alerts(alert)
            
            # Send email alerts
            if os.getenv("ENABLE_EMAIL_ALERTS") == "true":
                self._send_email_alerts(alert)
            
            # Push to dashboard
            self._push_to_dashboard(alert)
            
            # Store alert in database
            self._store_alert(alert)
            
            # Update alert status
            alert["status"] = "sent"
            logger.info(f"Alert {alert['id']} sent successfully")
        
        return alerts
    
    def _send_sms_alerts(self, alert):
        """Send SMS alerts using Twilio"""
        sms_recipients = [
            contact["phone"] for contact in alert["recipients"] 
            if "phone" in contact and contact.get("channels", {}).get("sms", False)
        ]
        
        for recipient in sms_recipients:
            try:
                message = self.twilio_client.messages.create(
                    body=alert["message"],
                    from_=os.getenv("TWILIO_PHONE_NUMBER"),
                    to=recipient
                )
                
                logger.info(f"SMS alert sent to {recipient}, SID: {message.sid}")
            except Exception as e:
                logger.error(f"Failed to send SMS to {recipient}: {str(e)}")
    
    def _send_email_alerts(self, alert):
        """Send email alerts"""
        email_recipients = [
            contact["email"] for contact in alert["recipients"] 
            if "email" in contact and contact.get("channels", {}).get("email", False)
        ]
        
        # Placeholder for email sending logic
        # In a real implementation, this would use a service like SendGrid, AWS SES, etc.
        for recipient in email_recipients:
            try:
                # Placeholder for actual email sending
                logger.info(f"Email alert sent to {recipient}")
            except Exception as e:
                logger.error(f"Failed to send email to {recipient}: {str(e)}")
    
    def _push_to_dashboard(self, alert):
        """Push alert to real-time dashboard"""
        try:
            # Store in Redis for real-time access
            alert_key = f"alert:{alert['id']}"
            self.db_manager.redis_client.set(
                alert_key,
                json.dumps(alert),
                ex=86400  # Expire after 24 hours
            )
            
            # Add to recent alerts list
            self.db_manager.redis_client.lpush("recent_alerts", alert_key)
            self.db_manager.redis_client.ltrim("recent_alerts", 0, 99)  # Keep last 100 alerts
            
            logger.info(f"Alert {alert['id']} pushed to dashboard")
        except Exception as e:
            logger.error(f"Failed to push alert to dashboard: {str(e)}")
    
    def _store_alert(self, alert):
        """Store alert in MongoDB for historical records"""
        try:
            db = self.db_manager.mongo_client["healthcare_data"]
            collection = db["alerts"]
            
            result = collection.insert_one(alert)
            logger.info(f"Alert {alert['id']} stored in database with _id: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Failed to store alert in database: {str(e)}")
    
    def get_active_alerts(self):
        """Get all active alerts"""
        try:
            # Fetch active alerts from Redis
            alert_keys = self.db_manager.redis_client.lrange("recent_alerts", 0, -1)
            active_alerts = []
            
            for key in alert_keys:
                alert_json = self.db_manager.redis_client.get(key)
                if alert_json:
                    alert = json.loads(alert_json)
                    active_alerts.append(alert)
            
            logger.info(f"Retrieved {len(active_alerts)} active alerts")
            return active_alerts
        except Exception as e:
            logger.error(f"Failed to retrieve active alerts: {str(e)}")
            return []

# API and Web Interface
class APIService:
    def __init__(self, db_manager, data_pipeline, model_manager, alert_system):
        self.db_manager = db_manager
        self.data_pipeline = data_pipeline
        self.model_manager = model_manager
        self.alert_system = alert_system
        
        # Initialize scheduler for periodic tasks
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        
        # Schedule periodic prediction job
        self.scheduler.add_job(
            self._periodic_prediction_job,
            'cron',
            hour=os.getenv("PREDICTION_JOB_HOUR", "0"),
            minute=os.getenv("PREDICTION_JOB_MINUTE", "0")
        )
        
        # Setup API routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes"""
        @app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "timestamp": datetime.datetime.now().isoformat()})
        
        @app.route('/api/predictions', methods=['GET'])
        def get_predictions():
            region = request.args.get('region')
            days = int(request.args.get('days', 7))
            
            # Get predictions from database
            predictions = self._get_stored_predictions(region, days)
            
            return jsonify({"predictions": predictions})
        
        @app.route('/api/alerts', methods=['GET'])
        def get_alerts():
            region = request.args.get('region')
            active_only = request.args.get('active_only', 'true').lower() == 'true'
            
            # Get alerts
            if active_only:
                alerts = self.alert_system.get_active_alerts()
            else:
                alerts = self._get_historical_alerts(region)
            
            return jsonify({"alerts": alerts})
        
        @app.route('/api/predict', methods=['POST'])
        def run_prediction():
            data = request.json
            
            # Validate request
            if not data or not data.get('region_ids'):
                return jsonify({"error": "Invalid request. Must include region_ids"}), 400
            
            # Run prediction
            result = self._run_prediction_job(
                data.get('region_ids'),
                data.get('start_date'),
                data.get('end_date'),
                data.get('model_type', 'lstm')
            )
            
            return jsonify(result)
        
        @app.route('/api/dashboard/stats', methods=['GET'])
        def get_dashboard_stats():
            region = request.args.get('region')
            
            # Fetch dashboard statistics
            stats = self._get_dashboard_statistics(region)
            
            return jsonify(stats)
        
        logger.info("API routes set up successfully")
    
    def _periodic_prediction_job(self):
        """Run prediction job on a schedule"""
        logger.info("Starting scheduled prediction job")
        
        try:
            # Get all regions
            regions = self._get_all_regions()
            
            # Calculate date range for analysis
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)
            
            # Run prediction for all regions
            self._run_prediction_job(
                regions,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                "lstm"
            )
            
            logger.info("Scheduled prediction job completed successfully")
        except Exception as e:
            logger.error(f"Scheduled prediction job failed: {str(e)}")
    
    def _get_all_regions(self):
        """Get all regions from database"""
        try:
            db = self.db_manager.mongo_client["healthcare_data"]
            collection = db["regions"]
            
            regions = collection.distinct("region_id")
            return regions
        except Exception as e:
            logger.error(f"Failed to fetch regions: {str(e)}")
            return []
    
    def _run_prediction_job(self, region_ids, start_date, end_date, model_type):
        """Run a prediction job for specified regions"""
        logger.info(f"Running prediction job for {len(region_ids)} regions using {model_type} model")
        
        try:
            # Fetch and preprocess data
            data = self.data_pipeline.fetch_and_merge_data(region_ids, start_date, end_date)
            
            if data.empty:
                logger.error("No data available for prediction")
                return {"error": "No data available for prediction"}
            
            # Create target variable (placeholder - in real system this would be based on actual outbreak data)
            # Here we're creating a synthetic target for demonstration
            if 'admission_count' in data.columns:
                # Using admission count threshold as a proxy for outbreak
                threshold = data['admission_count'].mean() + 1.5 * data['admission_count'].std()
                data['outbreak'] = (data['admission_count'] > threshold).astype(int)
            else:
                # Synthetic target based on temperature and humidity
                if 'temperature' in data.columns and 'humidity' in data.columns:
                    # Higher risk with high temperature and high humidity
                    data['risk_score'] = (
                        (data['temperature'] - data['temperature'].min()) / 
                        (data['temperature'].max() - data['temperature'].min())
                    ) * (
                        (data['humidity'] - data['humidity'].min()) /
                        (data['humidity'].max() - data['humidity'].min())
                    )
                    
                    data['outbreak'] = (data['risk_score'] > 0.7).astype(int)
                else:
                    # Random synthetic target for demonstration
                    data['outbreak'] = np.random.binomial(1, 0.1, size=len(data))
            
            # Preprocess data
            processed_data = self.data_pipeline.preprocess_data(data)
            
            # Split features and target
            X = processed_data.drop(columns=['outbreak'])
            y = processed_data['outbreak']
            
            # If model hasn't been trained yet, train it
            if model_type not in self.model_manager.models:
                # Split data for training
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                self.model_manager.train_model(model_type, X_train, y_train)
                
                # Evaluate model
                self.model_manager.evaluate_model(model_type, self.model_manager.models[model_type], X_test, y_test)
            
            # Make predictions
            predictions = self.model_manager.predict_outbreak(model_type, X)
            
            # Get region data for alerts
            region_data = self._get_region_data(region_ids)
            
            # Interpret predictions
            insights = self.model_manager.interpret_predictions(
                predictions,
                data['region_id'].values if 'region_id' in data.columns else region_ids,
                data['date'].values if 'date' in data.columns else []
            )
            
            # Generate and send alerts
            alerts = self.alert_system.generate_alert(insights, region_data)
            self.alert_system.send_alerts(alerts)
            
            # Store predictions
            self._store_predictions(predictions, insights, alerts)
            
            result = {
                "success": True,
                "predictions": predictions,
                "insights": insights,
                "alerts": alerts
            }
            
            logger.info("Prediction job completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Prediction job failed: {str(e)}")
            return {"error": str(e)}
    
    def _get_region_data(self, region_ids):
        """Get region data including contact information"""
        try:
            db = self.db_manager.mongo_client["healthcare_data"]
            collection = db["regions"]
            
            region_data = {}
            for region_id in region_ids:
                doc = collection.find_one({"region_id": region_id})
                if doc:
                    region_data[region_id] = doc
            
            return region_data
        except Exception as e:
            logger.error(f"Failed to fetch region data: {str(e)}")
            return {}
    
    def _store_predictions(self, predictions, insights, alerts):
        """Store prediction results in the database"""
        try:
            # Create document to store
            document = {
                "timestamp": datetime.datetime.now().isoformat(),
                "predictions": predictions,
                "insights": insights,
                "alerts": [alert["id"] for alert in alerts],
                "model_type": predictions.get("model_type")
            }
            
            # Store in MongoDB
            db = self.db_manager.mongo_client["healthcare_data"]
            collection = db["prediction_results"]
            
            result = collection.insert_one(document)
            logger.info(f"Prediction results stored in database with _id: {result.inserted_id}")
            
            # Store in Elasticsearch for analysis
            self.db_manager.store_prediction_results(document)
            
        except Exception as e:
            logger.error(f"Failed to store prediction results: {str(e)}")
    
    def _get_stored_predictions(self, region=None, days=7):
        """Get stored predictions from database"""
        try:
            db = self.db_manager.mongo_client["healthcare_data"]
            collection = db["prediction_results"]
            
            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Build query
            query = {
                "timestamp": {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            }
            
            if region:
                query["predictions.region"] = region
            
            # Fetch results
            cursor = collection.find(query).sort("timestamp", -1)
            predictions = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for pred in predictions:
                pred["_id"] = str(pred["_id"])
            
            logger.info(f"Retrieved {len(predictions)} prediction records")
            return predictions
        
        except Exception as e:
            logger.error(f"Failed to retrieve stored predictions: {str(e)}")
            return []
    
    def _get_historical_alerts(self, region=None):
        """Get historical alerts from database"""
        try:
            db = self.db_manager.mongo_client["healthcare_data"]
            collection = db["alerts"]
            
            # Build query
            query = {}
            if region:
                query["region"] = region
            
            # Fetch results
            cursor = collection.find(query).sort("timestamp", -1).limit(100)
            alerts = list(cursor)
            
            # Convert ObjectId to string for JSON serialization
            for alert in alerts:
                alert["_id"] = str(alert["_id"])
            
            logger.info(f"Retrieved {len(alerts)} historical alerts")
            return alerts
        
        except Exception as e:
            logger.error(f"Failed to retrieve historical alerts: {str(e)}")
            return []
    
    def _get_dashboard_statistics(self, region=None):
        """Get statistics for dashboard display"""
        try:
            # Get active alerts
            active_alerts = self.alert_system.get_active_alerts()
            
            # Filter by region if specified
            if region:
                active_alerts = [alert for alert in active_alerts if alert.get("region") == region]
            
            # Count alerts by risk level
            risk_counts = {
                "High": len([a for a in active_alerts if a.get("risk_level") == "High"]),
                "Medium": len([a for a in active_alerts if a.get("risk_level") == "Medium"]),
                "Low": len([a for a in active_alerts if a.get("risk_level") == "Low"]),
            }
            
            # Get recent predictions
            predictions = self._get_stored_predictions(region, days=7)
            
            # Summarize prediction trends
            prediction_trend = []
            if predictions:
                # Group by date and calculate average
                # This is a simplified example
                for pred in predictions[:7]:
                    date = pred.get("timestamp", "").split("T")[0]
                    avg_prob = 0
                    if "predictions" in pred and "outbreak_probability" in pred["predictions"]:
                        probs = pred["predictions"]["outbreak_probability"]
                        if probs:
                            avg_prob = sum(probs) / len(probs)
                    
                    prediction_trend.append({
                        "date": date,
                        "average_probability": avg_prob
                    })
            
            stats = {
                "active_alerts": len(active_alerts),
                "risk_level_counts": risk_counts,
                "prediction_trend": prediction_trend,
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            logger.info("Dashboard statistics generated successfully")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to generate dashboard statistics: {str(e)}")
            return {
                "error": str(e),
                "active_alerts": 0,
                "risk_level_counts": {"High": 0, "Medium": 0, "Low": 0},
                "prediction_trend": []
            }

# Main application
def main():
    """Main function to initialize and run the application"""
    try:
        # Initialize components
        db_manager = DatabaseManager()
        data_pipeline = DataPipeline(db_manager)
        model_manager = ModelManager(db_manager)
        alert_system = AlertSystem(db_manager)
        api_service = APIService(db_manager, data_pipeline, model_manager, alert_system)
        
        # Start the Flask application
        port = int(os.getenv("APP_PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=os.getenv("DEBUG", "false").lower() == "true")
        
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        sys.exit(1)

