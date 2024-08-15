from textSummarizer.config.configuration import ConfigurationManager  # Import ConfigurationManager to handle configuration files.
from textSummarizer.components.data_transformation import DataTransformation  # Import DataTransformation class for data transformation tasks.
from textSummarizer.logging import logger  # Import logger for logging information.

class DataTransformationTrainingPipeline:
    def __init__(self):
        """Initialize the DataTransformationTrainingPipeline class."""
        pass  # No initialization required for now.

    def main(self):
        """Main method to execute the data transformation pipeline."""
        # Initialize the configuration manager to read configuration settings.
        config = ConfigurationManager()
        
        # Retrieve the data transformation configuration from the configuration manager.
        data_transformation_config = config.get_data_transformation_config()
        
        # Initialize the DataTransformation class with the retrieved configuration.
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # Perform the data transformation process.
        data_transformation.convert()
