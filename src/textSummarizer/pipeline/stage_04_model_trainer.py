from textSummarizer.config.configuration import ConfigurationManager  # Import ConfigurationManager to handle configuration files.
from textSummarizer.components.model_trainer import ModelTrainer  # Import ModelTrainer class for training the model.
from textSummarizer.logging import logger  # Import logger for logging information.

class ModelTrainerTrainingPipeline:
    def __init__(self):
        """Initialize the ModelTrainerTrainingPipeline class."""
        pass  # No initialization required for now.

    def main(self):
        """Main method to execute the model training pipeline."""
        # Initialize the configuration manager to read configuration settings.
        config = ConfigurationManager()
        
        # Retrieve the model trainer configuration from the configuration manager.
        model_trainer_config = config.get_model_trainer_config()
        
        # Initialize the ModelTrainer class with the retrieved configuration.
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        # Perform the model training process.
        model_trainer.train()
