from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (MonitorInferenceLog, MonitorInferenceLogProblemType,
MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric)


class ModelMonitor:
    """ 
    A class for monitoring model post-training performance using Databricks Lakehouse Monitor.
    Attributes:
    baseline_table : str
        The name of the table containing baseline data for model performance comparison.
    inference_table : str
        The name of the table containing inference data for monitoring model performance.
    """
    def __init__(self, catalog, schema, baseline_table, inference_table, assets_dir):
        self.catalog = catalog,
        self.schema = schema,
        self.baseline_table = baseline_table
        self.inference_table = inference_table
        self.assets_dir = assets_dir

    def create_monitor(self):
        """ 
        Define the monitor configuration.
        """
        # define problem type
        problem_type = MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION
        # granularity of the monitor
        granularity = ["1 day"]
        # instantiate the monitor inference log
        inference_log = MonitorInferenceLog(
            timestamp_col="timestamp",
            granularities=granularity,
            problem_type=problem_type,
        )
        # connect to the workspace
        w = WorkspaceClient()
        # create a monitor
        info = w.quality_monitors.create(
            table_name=f"{self.catalog}.{self.schema}.{self.inference_table}",
            assets_dir=self.assets_dir,
            output_schema_name=f"{self.catalog}.{self.schema}",
            baseline_table_name=f"{self.catalog}.{self.schema}.{self.baseline_table}",
        )

    def update_monitor(self):
        """
        Update the monitor configuration.
        """
        # define problem type
        problem_type = MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION
        # granularity of the monitor
        granularity = ["1 day"]
        # asset directory for storing monitor artifacts
        asset_directory = self.assets_dir
        # instantiate the monitor inference log
        inference_log = MonitorInferenceLog(
            timestamp_col="timestamp",
            granularities=granularity,
            problem_type=problem_type,
            asset_directory=asset_directory
        )
        # connect to the workspace
        w = WorkspaceClient()
        # update the monitor
        info = w.quality_monitors

    def monitor(self):
        """
        Monitor model performance.
        """
        # connect to the workspace
        w = WorkspaceClient()
        try:
            # get the monitor info
            if w.quality_monitors.get(name=self.inference_table):
                print(f"Monitor {self.inference_table} already exists, updating the monitor...")
                # update the monitor
                w.quality_monitors.update(

                )

        except Exception as e:
            print(f"Monitor {self.inference_table} does not exist, creating the monitor...")
            # create the monitor
            w.quality_monitors.create(

            )