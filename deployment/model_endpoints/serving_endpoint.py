import mlflow
from mlflow import MlflowClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag, ServedEntityInput
from databricks.sdk import WorkspaceClient

class ModelDeployment:
    """
    Model Deployment class that encapsulates the deployment of a model to a serving endpoint.
    """
    def __init__(self, model_name, inference_table_name, model_alias, endpoint_name):
        self.model_name = model_name
        self.inference_table_name = inference_table_name
        self.model_alias = model_alias
        self.endpoint_name = endpoint_name
        self.mlflow_client = MlflowClient()

    def get_model_version(self):
        """ 
        Get the version of the model to be deployed
        """
        model_version = self.mlflow_client.get_model_version_by_alias(self.model_name, self.model_alias).version
        return model_version
    
    def endpoint_config_dict(self):
        """ 
        This method defines the endpoint configuration for the model to be deployed.
        This will be used when  new endpoint is created.
        """
        # parse model name from UC namespace
        served_model_name =  self.model_name.split('.')
        
        # define endpoint configuration
        endpoint_config_dict = {
            "served_models": [
                {
                    "model_name": self.model_name,
                    "model_version": self.get_model_version(),
                    "scale_to_zero_enabled": True,
                    "workload_size": "Small"
                }
            ],
            "traffic_config": {
                "routes": [
                    {"served_model_name": f"{served_model_name[-1]}-{self.get_model_version()}", "traffic_percentage": 100}
                ]
            },
            "auto_capture_config": {
                "catalog_name": served_model_name[0],
                "schema_name": served_model_name[1],
                "table_name_prefix": self.inference_table_name
            }
        }
        endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
        return endpoint_config
    
    def create_served_entities(self):
        """
        This config defines model to be served by the endpoint.
        This will be used when updating the existing endpoint.
        """
        served_entities = [
            ServedEntityInput.from_dict({
                "name": self.endpoint_name,
                "entity_name": self.model_name,
                "entity_version": self.get_model_version(),
                "scale_to_zero_enabled": True,
                "workload_size": "Small"
            })
        ]
        return served_entities
    
    def create_endpoint(self):
        """"
        Deploys a model serving endpoint.
        This method instantiates a workspace client and attempts to deploy a model to an endpoint.
        It first checks if the endpoint already exists, if it does, it updates the endpoint, otherwise it creates a new endpoint.
        """
        # instantiate workspace client
        w = WorkspaceClient()
        # deploy to endpoint
        try:
            # attempt to get the endpoint name
            if w.serving_endpoints.get(name=self.endpoint_name).name:
                print(f"Endpoint {self.endpoint_name} already exists, updating the endpoint...")
                w.serving_endpoints.update_config(
                    name=self.endpoint_name,
                    served_entities=self.create_served_entities()
                )
        except Exception as e:
            print(f"Endpoint {self.endpoint_name} does not exist, creating the endpoint...")
            w.serving_endpoints.create_and_wait(
                name=self.endpoint_name,
                config=self.endpoint_config_dict(),
                tags=[EndpointTag.from_dict({"key": "model_name", "value": self.model_name})]
            )
            print(f"Created endpoint {self.endpoint_name} with model {self.model_name} of version:{self.get_model_version()}")