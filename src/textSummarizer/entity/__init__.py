from dataclasses import dataclass  # Importing the dataclass decorator for defining data classes.
from pathlib import Path  # Importing Path for type hints related to file paths.

@dataclass(frozen=True)  # Decorator to create an immutable data class.
class DataIngestionConfig:
    root_dir: Path  # Directory where data ingestion artifacts will be stored.
    source_URL: str  # URL from which data will be downloaded.
    local_data_file: Path  # Local path where the downloaded data file will be saved.
    unzip_dir: Path  # Directory where the downloaded zip file will be extracted.

@dataclass(frozen=True)  # Decorator to create an immutable data class.
class DataValidationConfig:
    root_dir: Path  # Directory where validation artifacts will be stored.
    STATUS_FILE: str  # Path to the file that tracks the status of data validation.
    ALL_REQUIRED_FILES: list  # List of filenames that are required for validation.

@dataclass(frozen=True)  # Decorator to create an immutable data class.
class DataTransformationConfig:
    root_dir: Path  # Directory where transformed data will be stored.
    data_path: Path  # Path to the raw data directory for transformation.
    tokenizer_name: Path  # Path or name of the tokenizer to be used.

@dataclass(frozen=True)  # Decorator to create an immutable data class.
class ModelTrainerConfig:
    root_dir: Path  # Directory where model training artifacts will be stored.
    data_path: Path  # Path to the transformed data used for training.
    model_ckpt: Path  # Path to the pre-trained model checkpoint to be used.
    num_train_epochs: int  # Number of epochs for training.
    warmup_steps: int  # Number of steps for learning rate warmup.
    per_device_train_batch_size: int  # Batch size for training per device.
    weight_decay: float  # Weight decay for regularization.
    logging_steps: int  # Frequency of logging training metrics.
    evaluation_strategy: str  # Strategy for model evaluation.
    eval_steps: int  # Frequency of evaluation steps.
    save_steps: float  # Frequency of saving model checkpoints.
    gradient_accumulation_steps: int  # Number of steps to accumulate gradients before updating model weights.

@dataclass(frozen=True)  # Decorator to create an immutable data class.
class ModelEvaluationConfig:
    root_dir: Path  # Directory where evaluation results will be stored.
    data_path: Path  # Path to the data used for evaluation.
    model_path: Path  # Path to the trained model for evaluation.
    tokenizer_path: Path  # Path to the tokenizer used during training.
    metric_file_name: Path  # Path to the file where evaluation metrics will be saved.
