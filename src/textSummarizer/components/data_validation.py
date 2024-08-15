import os  # Importing os module for file system operations like listing files in a directory.
from textSummarizer.logging import logger  # Importing logger to handle logging messages.
from textSummarizer.entity import DataValidationConfig  # Importing DataValidationConfig data class for configuration.

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        """Initialize the DataValidation class with configuration.

        Args:
            config (DataValidationConfig): Configuration object for data validation.
        """
        self.config = config  # Store the configuration object.

    def validate_all_files_exist(self) -> bool:
        """Validate if all required files exist in the data directory.

        Returns:
            bool: True if all required files are present, False otherwise.
        """
        try:
            validation_status = None  # Initialize variable to track validation status.

            # List all files in the specified directory.
            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "samsum_dataset"))

            # Check if each required file is present in the directory.
            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False  # Set validation status to False if a required file is missing.
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")  # Write validation status to the status file.
                else:
                    validation_status = True  # Set validation status to True if the file is found.
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")  # Write validation status to the status file.

            return validation_status  # Return the final validation status.
        
        except Exception as e:
            # Handle exceptions and raise them for further handling.
            raise e
