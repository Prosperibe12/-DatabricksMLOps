from abc import ABC, abstractmethod

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
    def train_model(self, X_train, X_test, y_train, y_test):
        """ Abstract method to train the model. """
        raise NotImplementedError("Concrete class must implement this method")

    @abstractmethod
    def tune_hyperparameters(self, X_train, X_test, y_train, y_test):
        """ Abstract method for hyperparameter tuning. """
        raise NotImplementedError("Concrete class must implement this method")

    @abstractmethod
    def start(self):
        """ Abstract method to orchestrate the entire training process"""
        raise NotImplementedError("Concrete class must implement this method")