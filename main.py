from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline  # Import the data ingestion pipeline.
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline  # Import the data validation pipeline.
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline  # Import the data transformation pipeline.
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline  # Import the model trainer pipeline.
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline  # Import the model evaluation pipeline.
from textSummarizer.logging import logger  # Import the logger for logging information.

# Define and run the Data Ingestion stage.
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of the data ingestion stage.
    data_ingestion = DataIngestionTrainingPipeline()  # Create an instance of DataIngestionTrainingPipeline.
    data_ingestion.main()  # Execute the main method of DataIngestionTrainingPipeline to start the ingestion process.
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log the completion of the data ingestion stage.
except Exception as e:
    logger.exception(e)  # Log the exception if an error occurs during the data ingestion stage.
    raise e  # Re-raise the exception for further handling.

# Define and run the Data Validation stage.
STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of the data validation stage.
    data_validation = DataValidationTrainingPipeline()  # Create an instance of DataValidationTrainingPipeline.
    data_validation.main()  # Execute the main method of DataValidationTrainingPipeline to start the validation process.
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log the completion of the data validation stage.
except Exception as e:
    logger.exception(e)  # Log the exception if an error occurs during the data validation stage.
    raise e  # Re-raise the exception for further handling.

# Define and run the Data Transformation stage.
STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of the data transformation stage.
    data_transformation = DataTransformationTrainingPipeline()  # Create an instance of DataTransformationTrainingPipeline.
    data_transformation.main()  # Execute the main method of DataTransformationTrainingPipeline to start the transformation process.
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log the completion of the data transformation stage.
except Exception as e:
    logger.exception(e)  # Log the exception if an error occurs during the data transformation stage.
    raise e  # Re-raise the exception for further handling.

# Define and run the Model Trainer stage.
STAGE_NAME = "Model Trainer stage"
try:
    logger.info(f"*******************")  # Log a separator for clarity.
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of the model training stage.
    model_trainer = ModelTrainerTrainingPipeline()  # Create an instance of ModelTrainerTrainingPipeline.
    model_trainer.main()  # Execute the main method of ModelTrainerTrainingPipeline to start the training process.
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log the completion of the model training stage.
except Exception as e:
    logger.exception(e)  # Log the exception if an error occurs during the model training stage.
    raise e  # Re-raise the exception for further handling.

# Define and run the Model Evaluation stage.
STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f"*******************")  # Log a separator for clarity.
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")  # Log the start of the model evaluation stage.
    model_evaluation = ModelEvaluationTrainingPipeline()  # Create an instance of ModelEvaluationTrainingPipeline.
    model_evaluation.main()  # Execute the main method of ModelEvaluationTrainingPipeline to start the evaluation process.
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log the completion of the model evaluation stage.
except Exception as e:
    logger.exception(e)  # Log the exception if an error occurs during the model evaluation stage.
    raise e  # Re-raise the exception for further handling.
