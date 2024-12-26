import logging

import mlflow
import mlflow.exceptions
from mlflow.models import MetricThreshold
from pyspark.sql import SparkSession

class RegisterEvaluateModel:
    """ 
    This class provides utilities for registering the best run model in the model registry,
    performing model evaluation before setting the registered model tag to approved.
    """
    model_alias = "champion"

    def __init__(self, model_name, experiment_name, user, validation_data_path):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.current_user = user
        self.validation_data_path = validation_data_path
        self.client = mlflow.MlflowClient()
        self.spark = SparkSession.builder.getOrCreate()

    def register_model(self, run_id):
        """
        Register a model in the model registry using its run id.
        """
        try:
            uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=uri, 
                name=self.model_name
            )
            logging.info(f"Successfully registered model '{self.model_name}' with version {registered_model.version}.")
            return registered_model

        except Exception as e:
            logging.error("Invalid parameters provided for model registration.")
            raise ValueError(f"InvalidParameterException: {str(e)}")

    def evaluate_model(self, model_info):
        """
        Evaluate a model before deploying to production (setting the alias to champion).
        If a model with alias 'champion' exists, perform a baseline comparison.
        If no 'champion' exists, perform static evaluation with predefined thresholds.
        """
        # Load evaluation data
        df = self._load_validation_data()
        experiment = mlflow.get_experiment_by_name(f"/Users/{self.current_user}/{self.experiment_name}")

        try:
            # try to load the current champion model (baseline)
            baseline_model = self._load_baseline_model()

            if baseline_model:
                logging.info("Champion model found. Proceeding with baseline comparison.")

                # Perform baseline comparison
                try:
                    self._evaluate_with_baseline(df, model_info, experiment, baseline_model)

                    # Validate and assign aliases
                    self._assign_alias(
                        model_info=model_info,
                        promote_candidate=True
                    )

                except mlflow.exceptions.MlflowException as validation_error:
                    # Log failure and stop the process
                    logging.error(f"Baseline model validation failed: {validation_error}")
                    raise RuntimeError(f"Candidate model failed baseline validation with {validation_error}")

        except mlflow.exceptions.MlflowException:
            # This block executes only if no champion model is found
            logging.warning("No champion model found. Proceeding with static evaluation.")
            self._perform_static_evaluation(df, model_info, experiment)

    def _load_validation_data(self):
        """
        Load validation data as a Pandas DataFrame.
        """
        try:
            column_types = {
                "room_temperature": "float64",
                "room_light": "float64",
                "co2_ppm": "float64",
                "humidity_ratio": "float64",
                "Occupancy": "int64"
            }
            df = self.spark.read.format("csv").option("header", "true").load(self.validation_data_path).toPandas()
            data = df.astype(column_types)
            return data
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise ValueError(f"Validation data loading failed: {e}")

    def _load_baseline_model(self):
        """
        Load the baseline model with alias 'champion'.
        """
        baseline_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@{self.model_alias}")
        return baseline_model

    def _evaluate_with_baseline(self, df, model_info, experiment, baseline_model):
        """
        Evaluate the candidate model against the baseline (champion) model.
        """
        candidate_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{model_info.version}")

        # Evaluate the baseline model
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Occupancy_MLOps_Validation_Baseline") as baseline_run:
            baseline_eval = mlflow.evaluate(
                baseline_model,
                df,
                targets="Occupancy",
                model_type="classifier"
            )

        # Evaluate the candidate model
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Occupancy_MLOps_Validation_Candidate") as candidate_run:
            candidate_eval = mlflow.evaluate(
                candidate_model,
                df,
                targets="Occupancy",
                model_type="classifier"
            )

        # Define thresholds
        thresholds = {
            "accuracy_score": MetricThreshold(
                threshold=0.85,
                min_absolute_change=0.5,
                min_relative_change=0.05,
                greater_is_better=True,
            )
        }

        # Validate the candidate model against the baseline
        mlflow.validate_evaluation_results(
            validation_thresholds=thresholds,
            candidate_result=candidate_eval,
            baseline_result=baseline_eval
        )

    def _assign_alias(self, model_info, promote_candidate=False):
        """
        Validate the candidate model and assign aliases if successful.
        """
        try:
            baseline_model_version = self.client.get_model_version_by_alias(name=self.model_name, alias=self.model_alias)
            if promote_candidate:
                self.client.set_registered_model_alias(self.model_name, "previous", baseline_model_version.version)
                self.client.set_registered_model_alias(self.model_name, self.model_alias, model_info.version)
                logging.info(f"Candidate model {self.model_name}:{model_info.version} promoted to champion.")
        except Exception as e:
            logging.error(f"Assigning alias {self.model_alias} failed: {e}")
            raise RuntimeError(f"{self.model_name} with alias {self.model_alias} not found: {e}")

    def _perform_static_evaluation(self, df, model_info, experiment):
        """
        Perform static evaluation of the candidate model without a baseline comparison.
        """
        thresholds = {
            "accuracy_score": MetricThreshold(threshold=0.10, greater_is_better=True),
            "precision_score": MetricThreshold(threshold=0.10, greater_is_better=True),
            "recall_score": MetricThreshold(threshold=0.10, greater_is_better=True),
        }
        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{model_info.version}")
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Occupancy_MLOps_Validation_Static") as eval_run:
            candidate_result = mlflow.evaluate(
                model,
                df,
                targets="Occupancy",
                model_type="classifier"
            )
        mlflow.validate_evaluation_results(
            candidate_result=candidate_result,
            baseline_result=None,
            validation_thresholds=thresholds,
        )
        logging.info("Candidate model validated successfully against static thresholds.")
        self.client.set_registered_model_alias(self.model_name, self.model_alias, model_info.version)
        logging.info(f"Candidate model {self.model_name}:{model_info.version} set as champion.")