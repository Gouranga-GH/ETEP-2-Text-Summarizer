import os
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm

# Dataclass to store model evaluation configurations
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path

from textSummarizer.constants import *  # Import constants
from textSummarizer.utils.common import read_yaml, create_directories  # Import utility functions

# ConfigurationManager class for managing configurations
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name
        )

        return model_evaluation_config

# ModelEvaluation class for evaluating the model
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Splits the dataset into smaller batches for simultaneous processing."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text="dialogue", column_summary="summary"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, 
                               padding="max_length", return_tensors="pt")
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device), 
                                       length_penalty=0.8, num_beams=8, max_length=128)
            
            # Decode and process the generated summaries
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=True) 
                                 for s in summaries]      
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        # Compute and return the ROUGE scores
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = load_metric('rouge', trust_remote_code=True)

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][:10], rouge_metric, model_pegasus, tokenizer, 
            batch_size=2, column_text='dialogue', column_summary='summary'
        )

        # Prepare and save the ROUGE scores as a DataFrame
        rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)

# Main execution block
try:
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()
    model_evaluation = ModelEvaluation(config=model_evaluation_config)
    model_evaluation.evaluate()
except Exception as e:
    raise e
