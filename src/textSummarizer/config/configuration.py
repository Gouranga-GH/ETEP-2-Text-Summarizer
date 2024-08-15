from textSummarizer.constants import *  # Importing constants from the textSummarizer.constants module.
from textSummarizer.utils.common import read_yaml, create_directories  # Importing utility functions for reading YAML files and creating directories.
from textSummarizer.entity import (  # Importing data classes for configuration.
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,  # Default path for the configuration file.
        params_filepath = PARAMS_FILE_PATH  # Default path for the parameters file.
    ):
        # Read the YAML files for configuration and parameters.
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Create the root directory for storing artifacts as specified in the configuration.
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Retrieve and return data ingestion configuration."""
        config = self.config.data_ingestion  # Access the data ingestion configuration section.

        # Create the directory for data ingestion artifacts if it does not exist.
        create_directories([config.root_dir])

        # Create and return a DataIngestionConfig object with the retrieved settings.
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """Retrieve and return data validation configuration."""
        config = self.config.data_validation  # Access the data validation configuration section.

        # Create the directory for data validation artifacts if it does not exist.
        create_directories([config.root_dir])

        # Create and return a DataValidationConfig object with the retrieved settings.
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Retrieve and return data transformation configuration."""
        config = self.config.data_transformation  # Access the data transformation configuration section.

        # Create the directory for data transformation artifacts if it does not exist.
        create_directories([config.root_dir])

        # Create and return a DataTransformationConfig object with the retrieved settings.
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Retrieve and return model trainer configuration."""
        config = self.config.model_trainer  # Access the model trainer configuration section.
        params = self.params.TrainingArguments  # Access training arguments from parameters.

        # Create the directory for model training artifacts if it does not exist.
        create_directories([config.root_dir])

        # Create and return a ModelTrainerConfig object with the retrieved settings and parameters.
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            evaluation_strategy=params.evaluation_strategy,
            eval_steps=params.eval_steps,  # Fixed the assignment (was using evaluation_strategy instead of eval_steps).
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Retrieve and return model evaluation configuration."""
        config = self.config.model_evaluation  # Access the model evaluation configuration section.

        # Create the directory for model evaluation artifacts if it does not exist.
        create_directories([config.root_dir])

        # Create and return a ModelEvaluationConfig object with the retrieved settings.
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name
        )
        return model_evaluation_config
