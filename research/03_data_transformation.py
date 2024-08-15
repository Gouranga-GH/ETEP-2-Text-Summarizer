import os
from dataclasses import dataclass
from pathlib import Path

# Dataclass to define the configuration for data transformation
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

# Import necessary modules and constants from the project
from textSummarizer.constants import *  # Import project constants
from textSummarizer.utils.common import read_yaml, create_directories  # Import utility functions for reading YAML and creating directories

# ConfigurationManager class to manage reading and setting up the configuration
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)  # Read the main configuration file (YAML)
        self.params = read_yaml(params_filepath)  # Read the parameters file (YAML)
        create_directories([self.config.artifacts_root])  # Create the root directory for artifacts if it doesn't exist

    # Method to get the DataTransformationConfig object with all necessary paths and settings
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])  # Create the root directory for data transformation if it doesn't exist

        # Return the configuration as a DataTransformationConfig object
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )
        return data_transformation_config

# Import logging and other necessary modules for data transformation
from textSummarizer.logging import logger  # Import logger for logging activities
from transformers import AutoTokenizer  # Import the tokenizer from Hugging Face's transformers library
from datasets import load_dataset, load_from_disk  # Import dataset handling utilities

# DataTransformation class to handle the transformation of data
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)  # Load the tokenizer using the provided tokenizer name

    # Method to convert each example into tokenized features
    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'], max_length=1024, truncation=True
        )  # Tokenize the dialogue text

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'], max_length=128, truncation=True
            )  # Tokenize the summary text

        # Return the tokenized inputs, attention masks, and labels
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    # Method to perform the overall data transformation
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)  # Load the dataset from disk
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features, batched=True
        )  # Apply the tokenization process to the entire dataset
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )  # Save the processed dataset back to disk

# Main block to execute the data transformation process
try:
    config = ConfigurationManager()  # Initialize configuration manager
    data_transformation_config = config.get_data_transformation_config()  # Get the data transformation configuration
    data_transformation = DataTransformation(config=data_transformation_config)  # Initialize the DataTransformation class
    data_transformation.convert()  # Perform the data transformation
except Exception as e:
    raise e  # Handle exceptions
