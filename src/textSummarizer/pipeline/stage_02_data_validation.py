from textSummarizer.config.configuration import ConfigurationManager  # Import ConfigurationManager to manage configurations.
from textSummarizer.components.data_validation import DataValiadtion  # Import DataValiadtion class to handle data validation.
from textSummarizer.logging import logger  # Import logger for logging messages.

class DataValidationTrainingPipeline:
    def __init__(self):
        """Initialize the DataValidationTrainingPipeline class."""
        pass  # No initialization required for now.

    def main(self):
        """Main method to execute the data validation pipeline."""
        # Initialize the configuration manager to read configuration files.
        config = ConfigurationManager()
        
        # Retrieve the data validation configuration from the configuration manager.
        data_validation_config = config.get_data_validation_config()
        
        # Initialize the DataValiadtion class with the retrieved configuration.
        data_validation = DataValiadtion(config=data_validation_config)
        
        # Validate the existence of all required files.
        data_validation.validate_all_files_exist()
