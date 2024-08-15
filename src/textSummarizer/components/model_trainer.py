from transformers import TrainingArguments, Trainer  # Importing classes to define training arguments and manage the training loop.
from transformers import DataCollatorForSeq2Seq  # Importing data collator for sequence-to-sequence models.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # Importing pre-trained model and tokenizer classes.
from datasets import load_dataset, load_from_disk  # Importing functions to load datasets.
from textSummarizer.entity import ModelTrainerConfig  # Importing configuration data class for model training.
import torch  # Importing PyTorch for managing device placement (CPU/GPU).
import os  # Importing os module for file system operations.

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """Initialize the ModelTrainer class with configuration.

        Args:
            config (ModelTrainerConfig): Configuration object for model training.
        """
        self.config = config  # Store the configuration object.

    def train(self):
        """Train the model using the specified configuration."""
        # Set the device to GPU if available, otherwise use CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the tokenizer and model from the checkpoint specified in the configuration.
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        
        # Create a data collator for sequence-to-sequence models using the tokenizer and model.
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Load the dataset from the disk using the path specified in the configuration.
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Define training arguments, either using the provided configuration or hard-coded values.
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        ) 

        # Initialize the Trainer with the model, arguments, tokenizer, data collator, and datasets.
        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],  # Using test data for training to speed up the process.
            eval_dataset=dataset_samsum_pt["validation"]  # Validation dataset for evaluating the model.
        )
        
        # Start the training process.
        trainer.train()

        # Save the trained model to the specified directory.
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        # Save the tokenizer to the specified directory.
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
