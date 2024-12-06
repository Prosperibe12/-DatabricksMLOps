from _future_ import annotations
from abc import ABC, abstractmethod
import logging

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import mlflow
from mlflow.models.signature import infer_signature

from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import SparkTrials, fmin, tpe

from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Abstract Base Class for Machine Learning Models
class AbstractModelFactory(ABC):
    """ 
    Abstract Base Class for creating model factories.
    Ensures a consistent interface for data ingestion, preparation, training, and hyperparameter tuning.
    """
    @abstractmethod
    def get_data(self):
        """ Abstract method for data ingestion from a specified path. """
        raise NotImplementedError("Concrete class must implement this method")

    @abstractmethod
    def prepare_data(self, feature_table):
        """ Abstract method for preparing training data from raw data. """
        raise NotImplementedError("Concrete class must implement this method")

    @abstractmethod
    def split_normalize_data(self, data):
        """ Abstract method to split and normalize the dataset. """
        raise NotImplementedError("Concrete class must implement this method")

    @abstractmethod
    def train_model(self, data):
        """ Abstract method to train the model. """
        raise NotImplementedError("Concrete class must implement this method")

    @abstractmethod
    def tune_hyperparameters(self, model, data, hyperparameter_grid):
        """ Abstract method for hyperparameter tuning. """
        raise NotImplementedError("Concrete class must implement this method")

    @abstractmethod
    def start(self, path: str):
        """ Abstract method to start the model training process. """
        raise NotImplementedError("Concrete class must implement this method")

class ConcreteDecisionTreeModel(AbstractModelFactory):
    """
    Concrete class for Tensorflow model implementation
    """
    def __init__(self, feature_table: str, experiment_name: str, run_name: str):
        self.feature_table = feature_table
        self.experiment_name = experiment_name 
        self.run_name = run_name
        # set databricks unity catalog as model registry
        mlflow.set_registry_uri("databricks-uc")
        # set expriment name
        mlflow.set_experiment(self.experiment_name)
    
    def get_data(self):
        """
        Retrieve a dataset from the Feature Store using the specified path.
        Args:
            path (str): The Delta table path to load data from.
        Returns:
            DataFrame: Spark DataFrame containing the data.
        """
        try:
            data = spark.read.format("delta").load(self.feature_table)
            logging.info(f"Data successfully loaded from {self.feature_table}")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {self.feature_table}: {e}")
            raise RuntimeError(f"Data loading failed: {e}")

    def prepare_data(self, df):
        """
        Prepare the dataset for model training using features from the Feature Store.
        Args:
            df (DataFrame): Raw Spark DataFrame.
            feature_table (str): Feature Store table name.
        Returns:
            pd.DataFrame: Prepared Pandas DataFrame with features and labels.
        """
        try:
            fe = FeatureEngineeringClient()
            feature_lookup = [
                FeatureLookup(
                    table_name=self.feature_table,
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
            training_dataset = fe.create_training_set(
                df=df,
                feature_lookups=feature_lookup,
                label="Occupancy",
                exclude_columns=["Id", "Humidity"]
            )
            training_df = training_dataset.load_df()
            features_and_label = ["room_temperature", "room_light", "co2_ppm", "humidity_ratio", "Occupancy"]
            return training_df.toPandas()[features_and_label]
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise RuntimeError(f"Data preparation failed: {e}")

    def split_normalize_data(self, data: pd.DataFrame):
        """
        Split the data into training and testing sets and normalize the features.
        Args:
            data (pd.DataFrame): Input data with features and labels.
        Returns:
            tuple: Split and normalized (X_train, X_test, y_train, y_test).
        """
        try:
            X = data.drop(columns=["Occupancy"])
            y = data["Occupancy"]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            logging.info("Data split and normalized successfully.")

            return X_train, X_test, y_train, y_test
        
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

            # evaluate on the test set
            y_pred = dtc_mdl.predict(X_test)
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
            mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

            # Computing the confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

            # Creating a figure object and axes for the confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plotting the confusion matrix using the created axes
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
            disp.plot(cmap=plt.cm.Blues, ax=ax)

            # Setting the title of the plot
            ax.set_title('Confusion Matrix')

            # log confusion matrix
            mlflow.log_figure(figure=fig, artifact_file="confusion_matrix.png")

        return run
    
    def tune_hyperparameters(self, X_train, X_test, y_train, y_test):
        """
        Perform hyperparametre tunning for model using hyperopt framework
        """
        # retrieve experiment so they are logged together with the hyper parameter runs
        experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

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

                # evaluate the model on the test set
                y_pred = dtc_mdl.predict(X_test)
                accuracy_score(y_test, y_pred)
                precision_score(y_test, y_pred)
                recall_score(y_test, y_pred)
                f1_score(y_test, y_pred)

                # return the negative of our cross-validated F1 score as the loss
                return {
                "loss": -cv_score_results['f1'],
                "status": STATUS_OK,
                "run": mlflow_run
                }

        # set spark trials for parallel runs
        trials = SparkTrials(parallelism=4)
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{self.run_name}_Hyperparameter_Tunning") as parent_run:
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
            df = self.get_data(self.path)
            # do feature lookup from feature store
            data = self.prepare_data(df)
            # split and normalize data
            X_train, X_test, y_train, y_test = self.split_normalize_data(data)
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

# Factory for Creating Model Instances
class ConcreteModelFactory:
    """
    Implements the Concrete model class
    """
    
    def _init_(self, model_type: str, feature_table: str, experiment_name: str, run_name: str):
        self.model_type = model_type
        self.feature_table = feature_table
        self.experiment_name = experiment_name
        self.run_name = run_name

    def create_model(self):
        """
        Returns the appropriate model based on the specified type.
        """
        if self.model_type == "DecisionTree":
            return ConcreteDecisionTreeModel(self.feature_table, self.experiment_name, self.run_name)
        else:
            logging.error(f"Unsupported model type: {self.model_type}")
            raise ValueError(f"Unsupported model type: {self.model_type}")