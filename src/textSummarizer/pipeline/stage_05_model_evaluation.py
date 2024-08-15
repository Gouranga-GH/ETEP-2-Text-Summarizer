from textSummarizer.config.configuration import ConfigurationManager  # Import ConfigurationManager to handle configuration files.
from textSummarizer.components.model_evaluation import ModelEvaluation  # Import ModelEvaluation class for evaluating the model.
from textSummarizer.logging import logger  # Import logger for logging information.

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        """Initialize the ModelEvaluationTrainingPipeline class."""
        pass  # No initialization required for now.

    def main(self):
        """Main method to execute the model evaluation pipeline."""
        # Initialize the configuration manager to read configuration settings.
        config = ConfigurationManager()
        
        # Retrieve the model evaluation configuration from the configuration manager.
        model_evaluation_config = config.get_model_evaluation_config()
        
        # Initialize the ModelEvaluation class with the retrieved configuration.
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        # Perform the model evaluation process.
        model_evaluation.evaluate()
