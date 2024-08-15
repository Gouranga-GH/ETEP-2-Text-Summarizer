import os  # Importing os module for operating system related functions, such as checking file existence and creating directories.
import urllib.request as request  # Importing request from urllib for downloading files.
import zipfile  # Importing zipfile for handling zip file extraction.
from textSummarizer.logging import logger  # Importing logger for logging messages.
from textSummarizer.utils.common import get_size  # Importing utility function to get file size.
from pathlib import Path  # Importing Path for handling file system paths.
from textSummarizer.entity import DataIngestionConfig  # Importing DataIngestionConfig data class for configuration.

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """Initialize the DataIngestion class with configuration.

        Args:
            config (DataIngestionConfig): Configuration object for data ingestion.
        """
        self.config = config  # Store the configuration object.

    def download_file(self):
        """Download the file from the source URL to the local path if it does not already exist."""
        if not os.path.exists(self.config.local_data_file):
            # If the file does not exist, download it.
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! With the following info: \n{headers}")  # Log download success with file info.
        else:
            # If the file already exists, log its size.
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """Extract the zip file to the specified directory.

        Extracts the contents of the zip file into the data directory specified in the config.
        Function returns None.
        """
        unzip_path = self.config.unzip_dir  # Directory where the zip file will be extracted.
        os.makedirs(unzip_path, exist_ok=True)  # Create the directory if it does not exist.
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            # Open the zip file and extract its contents.
            zip_ref.extractall(unzip_path)
