import os  # Importing the os module for interacting with the operating system.
from box.exceptions import BoxValueError  # Importing BoxValueError for handling specific exceptions from the Box library.
import yaml  # Importing the yaml module for parsing YAML files.
from textSummarizer.logging import logger  # Importing the logger object for logging messages.
from ensure import ensure_annotations  # Importing ensure_annotations for type annotations checking.
from box import ConfigBox  # Importing ConfigBox for working with configuration files as dictionaries.
from pathlib import Path  # Importing Path from pathlib for handling file system paths.
from typing import Any  # Importing Any for flexible type annotations.

@ensure_annotations  # Decorator to ensure that function annotations are checked at runtime.
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If YAML file is empty.
        e: Other exceptions related to file reading.

    Returns:
        ConfigBox: Content of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:  # Open the YAML file for reading.
            content = yaml.safe_load(yaml_file)  # Load the content of the YAML file.
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")  # Log successful loading of the file.
            return ConfigBox(content)  # Return the content as a ConfigBox object.
    except BoxValueError:  # Handle specific error if the YAML file is empty.
        raise ValueError("yaml file is empty")  # Raise a ValueError with a specific message.
    except Exception as e:  # Handle other exceptions.
        raise e  # Re-raise the exception.

@ensure_annotations  # Decorator to ensure that function annotations are checked at runtime.
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories.

    Args:
        path_to_directories (list): List of paths to directories to be created.
        verbose (bool, optional): If True, logs each directory creation. Defaults to True.
    """
    for path in path_to_directories:  # Iterate over each path in the list.
        os.makedirs(path, exist_ok=True)  # Create the directory, including any necessary parent directories. Ignores if the directory already exists.
        if verbose:  # If verbose logging is enabled.
            logger.info(f"created directory at: {path}")  # Log the creation of the directory.

@ensure_annotations  # Decorator to ensure that function annotations are checked at runtime.
def get_size(path: Path) -> str:
    """Gets the size of a file in kilobytes.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size of the file in kilobytes.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)  # Calculate the file size in kilobytes.
    return f"~ {size_in_kb} KB"  # Return the size formatted as a string in kilobytes.
