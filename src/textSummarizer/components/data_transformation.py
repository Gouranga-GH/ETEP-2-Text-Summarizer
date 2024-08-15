import os  # Importing os module for file system operations.
from textSummarizer.logging import logger  # Importing logger for logging messages.
from transformers import AutoTokenizer  # Importing AutoTokenizer for tokenizing text using pre-trained models.
from datasets import load_dataset, load_from_disk  # Importing functions to load datasets.
from textSummarizer.entity import DataTransformationConfig  # Importing DataTransformationConfig data class for configuration.

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """Initialize the DataTransformation class with configuration.

        Args:
            config (DataTransformationConfig): Configuration object for data transformation.
        """
        self.config = config  # Store the configuration object.
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)  # Load the tokenizer based on the configuration.

    def convert_examples_to_features(self, example_batch):
        """Convert examples to features using the tokenizer.

        Args:
            example_batch (dict): Batch of examples containing 'dialogue' and 'summary'.

        Returns:
            dict: A dictionary containing tokenized input and target features.
        """
        # Tokenize the 'dialogue' field with a maximum length and truncation.
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

        # Use the tokenizer to tokenize the 'summary' field with a maximum length and truncation.
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)
        
        # Return a dictionary of tokenized input IDs, attention masks, and target labels.
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        """Load dataset from disk, convert it to features, and save it back to disk."""
        # Load the dataset from the specified path in the configuration.
        dataset_samsum = load_from_disk(self.config.data_path)

        # Apply the conversion function to each batch in the dataset.
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)

        # Save the transformed dataset to disk in the specified root directory.
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
