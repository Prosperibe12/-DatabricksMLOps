import logging

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.models.signature import infer_signature

from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import SparkTrials, fmin, tpe

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from .abstract_model import AbstractModelFactory
from .utils import log_confusion_matrix

class DecisionTreeModel(AbstractModelFactory):
    """
    Concrete implementation of a Decision Tree Model training pipeline.
    """
    def __init__(self, feature_table, experiment_name, validation_data_path, run_name, user):
        self.feature_table = feature_table
        self.experiment_name = experiment_name
        self.validation_data_path = validation_data_path
        self.run_name = run_name
        self.current_user = user
        self.spark = SparkSession.builder.getOrCreate()
        # set databricks unity catalog as model registry
        mlflow.set_registry_uri("databricks-uc")
        # set expriment name
        mlflow.set_experiment(f"/Users/{self.current_user}/{self.experiment_name}")
    
    def get_data(self):
        """
        Retrieve a dataset from the Feature Store using the specified path.
        """
        try:
            data = self.spark.read.format("delta").table(self.feature_table)
            logging.info(f"Data successfully loaded from {self.feature_table}")

            return data
        except Exception as e:
            logging.error(f"Error loading data from {self.feature_table}: {e}")
            raise RuntimeError(f"Data loading failed: {e}")

    def prepare_data(self, df):
        """
        Prepare the dataset for model training using features from the Feature Store.
        """
        try:
            # get feature table 
            feature_table = f"{self.feature_table}"
            
            # get lookup features
            feature_lookup = [
                FeatureLookup(
                    table_name=feature_table,
                    feature_names=["Temperature", "Light", "CO2", "HumidityRatio"],
                    lookup_key="Id",
                    rename_outputs={
                        "Temperature": "room_temperature",
                        "Light": "room_light",
                        "CO2": "co2_ppm",
                        "HumidityRatio": "humidity_ratio"
                    }
                )
            ]

            # create training set
            fe = FeatureEngineeringClient()

            # create training dataset
            training_dataset = fe.create_training_set(
                df=df,
                feature_lookups=feature_lookup,
                label="Occupancy",
                exclude_columns=["Id", "Humidity"]
            )

            # convert to pandas dataframe with selected features
            training_df = training_dataset.load_df()
            features_and_label = ["room_temperature", "room_light", "co2_ppm", "humidity_ratio", "Occupancy"]

            return training_df.toPandas()[features_and_label]
        
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise RuntimeError(f"Data preparation failed: {e}")

    def split_normalize_data(self, data):
        """
        Split the data into training and testing sets and normalize the features.
        """
        try:
            # shuffle the data
            data = data.sample(frac=1)
            # separate data into X & Y
            x = data.drop(columns=["Occupancy"])
            y = data["Occupancy"]

            # split data into train and test set
            X_data, X_test, y_data, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            # split further for validation set
            x_train,x_val,y_train,y_val = train_test_split(X_data,y_data, test_size=0.2,random_state=42)
            logging.info("Data split and normalized successfully.")

            # create a spark dataframe and save validation data
            x_val["Occupancy"] = y_val
            new_df = self.spark.createDataFrame(x_val)
            new_df.write.format("csv").option("header", "true").mode("overwrite").save(self.validation_data_path)

            # scale the data
            scaler = StandardScaler()
            scaled = scaler.fit(X_test)

            x_train_scaled = scaled.transform(x_train)
            x_test_scaled = scaled.transform(X_test)

            return x_train_scaled, x_test_scaled, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error in data splitting and normalization: {e}")
            raise RuntimeError(f"Data splitting and normalization failed: {e}")
        
    def train_model(self, X_train, X_test, y_train, y_test):
        """ 
        Perform model training using the specified algorithm
        """
        # turn off autologging
        mlflow.sklearn.autolog(disable=True)

        # define params for algorithm
        dtc_params = {
            'criterion': 'gini',
            'max_depth': 50,
            'min_samples_split': 20,
            'min_samples_leaf': 5
        }

        # start an MLFlow run
        with mlflow.start_run(run_name=self.run_name) as run:

            # log our parameters
            mlflow.log_params(dtc_params)

            # fit our model
            dtc = DecisionTreeClassifier(**dtc_params)
            dtc_mdl = dtc.fit(X_train, y_train)

            # define model signiture
            signature = infer_signature(X_train, y_train)

            # log the model
            mlflow.sklearn.log_model(
                sk_model = dtc_mdl, 
                artifact_path="model-artifacts",
                signature=signature,
            )

            # evaluate on the test set and log evaluation metrics
            y_pred = dtc_mdl.predict(X_test)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

            # log confusion matrix
            log_confusion_matrix(y_test, y_pred, labels=[1, 0], artifact_path="confusion_matrix.png")

        return run
    
    def tune_hyperparameters(self, X_train, X_test, y_train, y_test):
        """
        Perform hyperparametre tunning for model using hyperopt framework
        """
        # retrieve experiment so they are logged together with the hyper parameter runs
        experiment = mlflow.get_experiment_by_name(f"/Users/{self.current_user}/{self.experiment_name}")

        # define the search space
        search_space = {
            'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
            'max_depth': hp.choice('dtree_max_depth',
                                    [None, hp.uniformint('dtree_max_depth_int', 5, 50)]),
            'min_samples_split': hp.uniformint('dtree_min_samples_split', 2, 40),
            'min_samples_leaf': hp.uniformint('dtree_min_samples_leaf', 1, 20)
        }

        # define objective function
        def tuning_objective(params):
            # start the MLFlow run
            with mlflow.start_run(nested=True) as mlflow_run:
                # enable automatic logging of input samples, metrics, parameters, and models
                mlflow.sklearn.autolog(
                    disable=False,
                    log_input_examples=True,
                    silent=True,
                    exclusive=False)

                # set up model estimator
                dtc = DecisionTreeClassifier(**params)
                
                # cross-validated on the training set
                validation_scores = ['accuracy', 'precision', 'recall', 'f1']
                cv_results = cross_validate(dtc, 
                                            X_train, 
                                            y_train, 
                                            cv=5,
                                            scoring=validation_scores)
                # log the average cross-validated results
                cv_score_results = {}
                for val_score in validation_scores:
                    cv_score_results[val_score] = cv_results[f'test_{val_score}'].mean()
                    mlflow.log_metric(f"cv_{val_score}", cv_score_results[val_score])

                # fit the model on all training data
                dtc_mdl = dtc.fit(X_train, y_train)

                # return the negative of our cross-validated F1 score as the loss
                return {
                "loss": -cv_score_results['f1'],
                "status": STATUS_OK,
                "run": mlflow_run
                }

        # set spark trials for parallel runs
        trials = SparkTrials(parallelism=4)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{self.run_name}_Hyperparameter_Tunning") as parent_run:
            fmin(
                tuning_objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=5,
                trials=trials
            )
        # get the best run info
        best_result = trials.best_trial["result"]
        best_run = best_result["run"]

        return best_run.info.run_id

    def start(self):
        """
        Orchestrates the entire model training process.
        """
        try:
            # load data from feature store
            df = self.get_data()
            # do feature lookup from feature store
            prepared_data = self.prepare_data(df)
            # split and normalize data
            X_train, X_test, y_train, y_test = self.split_normalize_data(prepared_data)
            # train model
            train_run = self.train_model(X_train, X_test, y_train, y_test)
            # tune hyperparameter
            best_run = self.tune_hyperparameters(X_train, X_test, y_train, y_test)
    
            logging.info("Model Training and Hyper-parameter tunning completed!")
            # return best run id
            return best_run
        
        except Exception as e:
            logging.error(f"Error in the start process: {e}")
            raise RuntimeError(f"Model training process failed: {e}")