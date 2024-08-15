from textSummarizer.config.configuration import ConfigurationManager  # Import the ConfigurationManager to manage configurations.
from textSummarizer.components.data_ingestion import DataIngestion  # Import the DataIngestion class to handle data downloading and extraction.
from textSummarizer.logging import logger  # Import the logger for logging messages.

class DataIngestionTrainingPipeline:
    def __init__(self):
        """Initialize the DataIngestionTrainingPipeline class."""
        pass  # No initialization required for now.

    def main(self):
        """Main method to execute the data ingestion pipeline."""
        # Initialize the configuration manager to read configuration files.
        config = ConfigurationManager()
        
        # Retrieve the data ingestion configuration from the configuration manager.
        data_ingestion_config = config.get_data_ingestion_config()
        
        # Initialize the DataIngestion class with the retrieved configuration.
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # Download the data file from the specified URL if it does not exist.
        data_ingestion.download_file()
        
        # Extract the downloaded ZIP file into the specified directory.
        data_ingestion.extract_zip_file()
