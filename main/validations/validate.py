import logging 

import mlflow 
from mlflow.models import MetricThreshold
from pyspark.sql import SparkSession

class RegisterEvaluateModel:
    """ 
    This class provides utilities for registering the best run model in the model registry,
    perform model evaluation before setting the registered model tag to approved.
    """
    model_alias = "champion"
    
    def __init__(self, model_name, experiment_name, user,validation_data_path):
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
            # Create a new model using `mlflow.register_model`
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
        Evaluate a model before deploying to production (setting the alias to champion or production).
        If a model with alias 'champion' exists, perform a baseline comparison. 
        If no 'champion' exists, perform static evaluation with predefined thresholds.
        """
        # load evaluation data
        df = self._load_validation_data()
        # get training experiment
        experiment = mlflow.get_experiment_by_name(f"/Users/{self.current_user}/{self.experiment_name}")

        try:
            # Try to load the current champion model
            baseline_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@{self.model_alias}")
            logging.info("Champion model found. Proceeding with baseline comparison.")

            # Perform baseline comparison
            baseline_eval, candidate_eval = self._evaluate_with_baseline(df, model_info, experiment, baseline_model)

            # Validate and assign aliases
            self._validate_and_assign_alias(
                candidate_eval=candidate_eval,
                baseline_eval=baseline_eval,
                model_info=model_info,
                promote_candidate=True  # Promote candidate if validation succeeds
            )

        except mlflow.exceptions.MlflowException:
            logging.warning("No champion model found. Falling back to static evaluation.")

            # Perform static evaluation
            self._perform_static_evaluation(df, model_info, experiment)

    def _load_validation_data(self):
        """
        Load validation data as a Pandas DataFrame.
        """
        return self.spark.read.format("csv").option("header", "true").load(self.validation_data_path).toPandas()

    def _evaluate_with_baseline(self, df, model_info, experiment, baseline_model):
        """
        Evaluate the candidate model against the baseline (champion) model.
        """
        # load candidate model
        candidate_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{model_info.version}")

        # Evaluate the baseline model
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Occupancy_MLOps_Validation_Baseline") as baseline_run:
            baseline_eval = mlflow.evaluate(
                model=baseline_model,
                data=df,
                targets="Occupancy",
                model_type="classifier"
            )

        # Evaluate the candidate model
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Occupancy_MLOps_Validation_Candidate") as candidate_run:
            candidate_eval = mlflow.evaluate(
                model=candidate_model,
                data=df,
                targets="Occupancy",
                model_type="classifier"
            )

        return baseline_eval, candidate_eval

    def _validate_and_assign_alias(self, candidate_eval, baseline_eval, model_info, promote_candidate=False):
        """
        Validate the candidate model against the baseline and assign aliases if successful.
        """
        # define threshold change
        thresholds = {
            "accuracy_score": MetricThreshold(
                threshold=0.85,
                min_absolute_change=0.5,
                min_relative_change=0.05,
                greater_is_better=True,
            )
        }

        # Validate results
        mlflow.validate_evaluation_results(
            candidate_result=candidate_eval,
            baseline_result=baseline_eval,
            validation_thresholds=thresholds,
        )
        logging.info("Candidate model passed validation.")

        if promote_candidate:
            # get baseline model version
            baseline_model_version = self.client.get_model_version_by_alias(name=self.model_name, alias=self.model_alias)
            # Reassign aliases
            self.client.set_registered_model_alias(self.model_name, "previous", baseline_model_version.version)
            self.client.set_registered_model_alias(self.model_name, self.model_alias, model_info.version)
            logging.info(f"Candidate model {self.model_name}:{model_info.version} promoted to champion.")

    def _perform_static_evaluation(self, df, model_info, experiment):
        """
        Perform static evaluation of the candidate model without a baseline comparison.
        """
        # define metric thresholds
        thresholds = {
            "accuracy_score": MetricThreshold(threshold=0.85, greater_is_better=True),
            "precision_score": MetricThreshold(threshold=0.85, greater_is_better=True),
            "recall_score": MetricThreshold(threshold=0.85, greater_is_better=True),
        }
        # load the candidate model
        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{model_info.version}")
        # start model validation in a run
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Occupancy_MLOps_Validation_Static") as eval_run:
            candidate_result = mlflow.evaluate(
                model=model,
                data=df,
                targets="Occupancy",
                model_type="classifier"
            )
        mlflow.validate_evaluation_results(
            candidate_result=candidate_result,
            baseline_result=None,
            validation_thresholds=thresholds,
        )
        logging.info("Candidate model validated successfully against static thresholds.")

        # Assign alias to the candidate model
        self.client.set_registered_model_alias(self.model_name, self.model_alias, model_info.version)
        logging.info(f"Candidate model {self.model_name}:{model_info.version} set as champion.")