
# Text Summarization Web Application

This project implements a text summarization system using **FastAPI**, **Transformers**, and **Hugging Face** models. The system allows users to train, evaluate, and generate summaries using a pre-trained model like PEGASUS. The project is structured into multiple pipelines that handle data ingestion, validation, transformation, model training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Endpoints](#endpoints)
- [Model Pipeline Stages](#model-pipeline-stages)
- [Deployment](#deployment)
- [License](#license)

## Project Overview

This web application provides a platform to perform text summarization using state-of-the-art pre-trained models such as **PEGASUS**. It includes functionalities to train models, validate data, transform data for model training, and evaluate the trained models. The application uses **FastAPI** to expose a web interface and API.

## Features

- **Data Ingestion**: Automatically downloads and prepares the dataset.
- **Data Validation**: Ensures all required data files are available before processing.
- **Data Transformation**: Tokenizes text data and prepares it for model training.
- **Model Training**: Fine-tunes pre-trained models.
- **Model Evaluation**: Evaluates the model using the ROUGE metric.
- **Text Summarization**: Summarizes user-provided text using the trained model.
- **FastAPI Web Interface**: Simple web UI to interact with the summarization system.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/text-summarizer.git
cd text-summarizer
```

### 2. Set up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate
```

### 3. Install Dependencies
Install the required packages by running:
```bash
pip install -r requirements.txt
```

### 4. Docker (Optional)
If you want to run the application inside a Docker container, build the Docker image:
```bash
docker build -t text-summarizer-app .
```

## Running the Project

### 1. Run with FastAPI
To start the FastAPI server:
```bash
uvicorn app:app --reload
```
Navigate to `http://localhost:8000` in your browser.

### 2. Run with Docker
If using Docker:
```bash
docker run -p 8000:8000 text-summarizer-app
```



## Endpoints

### `/`
- **Description**: Home page of the application.
- **Method**: GET

### `/train`
- **Description**: Triggers the model training pipeline.
- **Method**: GET

### `/predict`
- **Description**: Allows users to input text and generate a summary.
- **Method**: POST

### `/docs`
- **Description**: API documentation (Swagger UI).
- **Method**: GET

## Model Pipeline Stages

1. **Data Ingestion (`stage_01_data_ingestion.py`)**: Downloads the dataset and extracts it.
2. **Data Validation (`stage_02_data_validation.py`)**: Ensures that all required files are available.
3. **Data Transformation (`stage_03_data_transformation.py`)**: Tokenizes text data for model training.
4. **Model Training (`stage_04_model_trainer.py`)**: Trains the PEGASUS model on the dataset.
5. **Model Evaluation (`stage_05_model_evaluation.py`)**: Evaluates the trained model using ROUGE.

## Deployment

### AWS CI/CD Deployment with GitHub Actions

#### Step-by-step Deployment Guide

1. **Login to AWS console** and create an IAM user with EC2 and ECR access.
2. **Build Docker image** from source code.
3. **Push Docker image** to ECR.
4. **Launch EC2 instance** (Ubuntu).
5. **Install Docker** on EC2 instance.
6. **Pull Docker image** from ECR on EC2 instance.
7. **Run Docker image** on EC2 instance.

## Required AWS Policies

- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

## Additional Steps

1. **Create ECR repository** and save the repository URI.
2. **Update and upgrade packages** on EC2 (optional).
3. **Configure EC2** as a self-hosted GitHub runner.
4. **Setup GitHub Secrets**:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `AWS_ECR_LOGIN_URI`
   - `ECR_REPOSITORY_NAME`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
