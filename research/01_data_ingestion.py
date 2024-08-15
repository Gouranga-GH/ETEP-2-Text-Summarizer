import os

# Change directory to one level up
os.chdir("../")

# Import necessary modules and classes
from dataclasses import dataclass
from pathlib import Path
from textSummarizer.constants import *  # Importing constants defined for the project
from textSummarizer.utils.common import read_yaml, create_directories  # Utility functions for reading YAML and creating directories

# Define a dataclass to hold data ingestion configuration
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

# ConfigurationManager class to manage reading and setting up the configuration
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)  # Read the configuration file (YAML)
        self.params = read_yaml(params_filepath)  # Read the parameters file (YAML)
        create_directories([self.config.artifacts_root])  # Create the artifacts root directory if it doesn't exist

    # Method to get the DataIngestionConfig object with all necessary paths and settings
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])  # Create the root directory for data ingestion

        # Return the configuration as a DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config

# Import additional modules for data ingestion
import urllib.request as request
import zipfile
from textSummarizer.logging import logger  # Import logger for logging activities
from textSummarizer.utils.common import get_size  # Utility function to get file size

# DataIngestion class to handle downloading and extracting the dataset
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # Method to download the dataset file if it doesn't already exist
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):  # Check if the file already exists
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,  # Download the file from the specified URL
                filename=self.config.local_data_file  # Save the file locally
            )
            logger.info(f"{filename} downloaded! with the following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    # Method to extract the contents of the zip file
    def extract_zip_file(self):
        """
        Extracts the zip file into the specified directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)  # Create the directory if it doesn't exist
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)  # Extract all files into the unzip directory

# Main block to execute the data ingestion process
try:
    config = ConfigurationManager()  # Initialize configuration manager
    data_ingestion_config = config.get_data_ingestion_config()  # Get the data ingestion configuration
    data_ingestion = DataIngestion(config=data_ingestion_config)  # Initialize the DataIngestion class
    data_ingestion.download_file()  # Download the dataset file
    data_ingestion.extract_zip_file()  # Extract the contents of the downloaded zip file
except Exception as e:
    raise e  # Handle exceptions that may occur during the process
