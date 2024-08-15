# Checking the GPU status
import os
os.system('!nvidia-smi')

# Installing necessary libraries for text summarization and evaluation
os.system('!pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q')
os.system('!pip install --upgrade accelerate')
os.system('!pip uninstall -y transformers accelerate')
os.system('!pip install transformers accelerate')

# Importing necessary libraries and modules
from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch

# Downloading the nltk tokenizer data
nltk.download("punkt")

# Setting the device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Defining the model checkpoint
model_ckpt = "google/pegasus-cnn_dailymail"

# Loading the tokenizer and model for PEGASUS
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

# Downloading the dataset from an external source
os.system('!wget https://github.com/Gouranga-GH/Additional-Files/raw/main/summarizer-data.zip')

# Unzipping the dataset
os.system('!unzip summarizer-data.zip')

# Loading the dataset from disk
dataset_samsum = load_from_disk('samsum_dataset')

# Displaying basic information about the dataset
split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum]
print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")

# Displaying a sample dialogue and its summary from the test set
print("\nDialogue:")
print(dataset_samsum["test"][1]["dialogue"])
print("\nSummary:")
print(dataset_samsum["test"][1]["summary"])

# Function to convert examples into features by tokenizing the input dialogue and summary
def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

# Applying the conversion function to the dataset and preparing it for PyTorch
dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
dataset_samsum_pt["train"]

# Importing modules required for model training
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

# Defining the data collator that handles padding and batching during training
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

# Defining the training arguments such as batch size, logging steps, etc.
trainer_args = TrainingArguments(
    output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10,
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16
)

# Creating a Trainer object to handle the training process
trainer = Trainer(
    model=model_pegasus, args=trainer_args,
    tokenizer=tokenizer, data_collator=seq2seq_data_collator,
    train_dataset=dataset_samsum_pt["train"],
    eval_dataset=dataset_samsum_pt["validation"]
)

# Starting the training process
trainer.train()

# Function to split a dataset into smaller batches
def generate_batch_sized_chunks(list_of_elements, batch_size):
    """Split the dataset into smaller batches that we can process simultaneously."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]

# Function to calculate ROUGE scores on the test dataset
def calculate_metric_on_test_ds(dataset, metric, model, tokenizer, batch_size=16, device=device, column_text="article", column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), length_penalty=0.8, num_beams=8, max_length=128)
        
        # Decoding the generated summaries and cleaning up the output
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
        
        # Adding predictions and references to the metric for calculation
        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    # Finally computing and returning the ROUGE scores
    score = metric.compute()
    return score

# Loading the ROUGE metric
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_metric = load_metric('rouge')

# Calculating ROUGE scores on the test dataset
score = calculate_metric_on_test_ds(
    dataset_samsum['test'], rouge_metric, trainer.model, tokenizer, batch_size=2, column_text='dialogue', column_summary='summary'
)

# Formatting and displaying the ROUGE scores
rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
pd.DataFrame(rouge_dict, index=[f'pegasus'])

# Saving the trained model and tokenizer for future use
model_pegasus.save_pretrained("pegasus-samsum-model")
tokenizer.save_pretrained("tokenizer")

# Loading the tokenizer for prediction
tokenizer = AutoTokenizer.from_pretrained("/content/tokenizer")

# Defining generation arguments for text summarization
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

# Extracting a sample dialogue and reference summary from the test set
sample_text = dataset_samsum["test"][0]["dialogue"]
reference = dataset_samsum["test"][0]["summary"]

# Setting up a pipeline for text summarization using the trained model and tokenizer
pipe = pipeline("summarization", model="pegasus-samsum-model", tokenizer=tokenizer)

# Displaying the sample dialogue, reference summary, and model-generated summary
print("Dialogue:")
print(sample_text)
print("\nReference Summary:")
print(reference)
print("\nModel Summary:")
print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])
