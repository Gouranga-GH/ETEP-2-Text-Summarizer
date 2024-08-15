import os
from dataclasses import dataclass
from pathlib import Path

# Dataclass to define the configuration for data validation
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

# Import necessary modules and constants from the project
from textSummarizer.constants import *  # Import project constants
from textSummarizer.utils.common import read_yaml, create_directories  # Import utility functions for reading YAML and creating directories

# ConfigurationManager class to manage reading and setting up the configuration
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)  # Read the main configuration file (YAML)
        self.params = read_yaml(params_filepath)  # Read the parameters file (YAML)
        create_directories([self.config.artifacts_root])  # Create the root directory for artifacts if it doesn't exist

    # Method to get the DataValidationConfig object with all necessary paths and settings
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])  # Create the root directory for data validation if it doesn't exist

        # Return the configuration as a DataValidationConfig object
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )
        return data_validation_config

# Import logging and other necessary modules for data validation
from textSummarizer.logging import logger  # Import logger for logging activities

# DataValidation class to handle the validation of required files
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    # Method to validate whether all required files exist
    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = None

            # List all files in the data ingestion directory
            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "samsum_dataset"))

            # Check if each required file is present in the directory
            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False  # Mark validation as failed if any file is missing
                    with open(self.config.STATUS_FILE, 'w') as f:  # Write validation status to a status file
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True  # Mark validation as passed if all files are present
                    with open(self.config.STATUS_FILE, 'w') as f:  # Write validation status to a status file
                        f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e  # Handle exceptions that may occur during the validation process

# Main block to execute the data validation process
try:
    config = ConfigurationManager()  # Initialize configuration manager
    data_validation_config = config.get_data_validation_config()  # Get the data validation configuration
    data_validation = DataValidation(config=data_validation_config)  # Initialize the DataValidation class
    data_validation.validate_all_files_exist()  # Perform file validation
except Exception as e:
    raise e  # Handle exceptions
