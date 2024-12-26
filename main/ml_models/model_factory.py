import logging
from main.ml_models.decision_tree_model import DecisionTreeModel


class ModelFactory:
    """
    Factory class to create model instances
    """
    
    def __init__(self, feature_table, experiment_name, validation_data_path, run_name, current_user):
        self.feature_table = feature_table
        self.experiment_name = experiment_name
        self.validation_data_path = validation_data_path
        self.run_name = run_name
        self.user = current_user

    def create_model(self, model_type):
        """
        Returns the appropriate model based on the specified type.
        """
        if model_type == "DecisionTree":
            return DecisionTreeModel(self.feature_table, self.experiment_name, self.validation_data_path, self.run_name, self.user)

        logging.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")