import os  # Importing the os module to interact with the operating system.
from pathlib import Path  # Importing Path from pathlib to handle file system paths.
import logging  # Importing logging for logging messages.

# Configuring the logging system to display information-level messages with a timestamp and message format.
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Defining the name of the project.
project_name = "textSummarizer"

# List of files and directories to be created for the project structure.
list_of_files = [
    ".github/workflows/.gitkeep",  # .gitkeep file to keep the empty workflows directory in version control.
    f"src/{project_name}/__init__.py",  # __init__.py file to mark the directory as a Python package.
    f"src/{project_name}/components/__init__.py",  # __init__.py for components sub-package.
    f"src/{project_name}/utils/__init__.py",  # __init__.py for utils sub-package.
    f"src/{project_name}/utils/common.py",  # A common.py file inside the utils package.
    f"src/{project_name}/logging/__init__.py",  # __init__.py for logging sub-package.
    f"src/{project_name}/config/__init__.py",  # __init__.py for config sub-package.
    f"src/{project_name}/config/configuration.py",  # configuration.py file for configuration settings.
    f"src/{project_name}/pipeline/__init__.py",  # __init__.py for pipeline sub-package.
    f"src/{project_name}/entity/__init__.py",  # __init__.py for entity sub-package.
    f"src/{project_name}/constants/__init__.py",  # __init__.py for constants sub-package.
    "config/config.yaml",  # YAML configuration file for general settings.
    "params.yaml",  # YAML file to store parameters for the project.
    "app.py",  # app.py file for the main application.
    "main.py",  # main.py file for running the project.
    "Dockerfile",  # Dockerfile to containerize the application.
    "requirements.txt",  # Text file to list the dependencies needed for the project.
    "setup.py",  # setup.py for packaging and distributing the project.
    "research/trials.ipynb",  # Jupyter notebook for research and experiments.
]

# Loop through each file path in the list to create the necessary directories and files.
for filepath in list_of_files:
    filepath = Path(filepath)  # Converting the file path into a Path object.
    filedir, filename = os.path.split(filepath)  # Splitting the file path into directory and filename.

    # If the directory path is not empty, create the directory if it does not exist.
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)  # Create the directory if it doesn't exist, ignoring any errors.
        logging.info(f"Creating directory: {filedir} for the file {filename}")  # Log that the directory was created.

    # Check if the file does not exist or if its size is zero, then create an empty file.
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:  # Open the file in write mode (creates an empty file).
            pass  # Placeholder for writing content (currently doing nothing).
        logging.info(f"Creating empty file: {filepath}")  # Log that an empty file was created.

    # If the file already exists and is not empty, log that it already exists.
    else:
        logging.info(f"{filename} already exists")
