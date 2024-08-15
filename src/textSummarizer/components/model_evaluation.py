from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # Import classes to load pre-trained model and tokenizer.
from datasets import load_dataset, load_from_disk, load_metric  # Import functions to load datasets and metrics.
import torch  # Import PyTorch for managing device placement (CPU/GPU).
import pandas as pd  # Import Pandas for handling data and saving results.
from tqdm import tqdm  # Import tqdm for progress bar visualization.
from textSummarizer.entity import ModelEvaluationConfig  # Import configuration data class for model evaluation.

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """Initialize the ModelEvaluation class with configuration.

        Args:
            config (ModelEvaluationConfig): Configuration object for model evaluation.
        """
        self.config = config  # Store the configuration object.

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches for processing.

        Args:
            list_of_elements (list): List of elements to split.
            batch_size (int): Size of each batch.

        Yields:
            list: Batch-sized chunks from list_of_elements.
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text="article", 
                                    column_summary="highlights"):
        """Calculate evaluation metrics on the test dataset.

        Args:
            dataset (Dataset): The dataset to evaluate.
            metric (Metric): The metric to compute.
            model (PreTrainedModel): The model to use for generating predictions.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for text processing.
            batch_size (int, optional): Batch size for processing. Defaults to 16.
            device (str, optional): Device to use (CUDA or CPU). Defaults to "cuda".
            column_text (str, optional): Column name for input text. Defaults to "article".
            column_summary (str, optional): Column name for target summaries. Defaults to "highlights".

        Returns:
            dict: The computed metric scores.
        """
        # Split the dataset into batches
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            # Tokenize the input text
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, 
                               padding="max_length", return_tensors="pt")
            
            # Generate summaries using the model
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device), 
                                       length_penalty=0.8, num_beams=8, max_length=128)
            ''' The length penalty prevents the model from generating overly long sequences. '''
            
            # Decode the generated summaries
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=True) 
                                 for s in summaries]
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            # Add the generated summaries and targets to the metric
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        # Compute and return the ROUGE scores
        score = metric.compute()
        return score

    def evaluate(self):
        """Evaluate the model and save the results."""
        # Set the device to GPU if available, otherwise use CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the tokenizer and model from the specified paths.
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        # Load the dataset from disk.
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Define the ROUGE metric names.
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
  
        # Load the ROUGE metric.
        rouge_metric = load_metric('rouge')

        # Calculate metrics on a subset of the test data (for demonstration, here only the first 10 examples are used).
        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer, 
            batch_size=2, column_text='dialogue', column_summary='summary'
        )

        # Create a dictionary of ROUGE scores.
        rouge_dict = {rn: score[rn].mid.fmeasure for rn in rouge_names}

        # Save the ROUGE scores to a CSV file.
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)
