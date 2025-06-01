# Student Performance Prediction - ML Project

## Overview
This project implements an end-to-end machine learning solution for predicting student math scores based on various demographic and academic factors. The system uses a web interface built with Flask and includes data preprocessing, model training, and deployment pipelines.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## Features
- Predicts student math scores based on:
  - Gender
  - Race/Ethnicity
  - Parental Education Level
  - Lunch Type
  - Test Preparation
  - Reading Scores
  - Writing Scores
- Web-based user interface
- Automated data preprocessing pipeline
- Model training and evaluation pipeline
- REST API endpoints for predictions

## Tech Stack
- Python 3.8+
- scikit-learn
- Flask
- pandas
- numpy
- Bootstrap 4
- HTML/CSS

## Project Structure
```
MLPROJECT/
│
├── artifacts/              # Trained models and preprocessors
├── logs/                  # Application logs
├── notebook/              # Jupyter notebooks for analysis
├── src/
│   ├── components/        # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/          # Training and prediction pipelines
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── utils.py          # Utility functions
│   ├── logger.py         # Logging configuration
│   └── exception.py      # Custom exception handling
├── templates/             # Flask HTML templates
│   ├── home.html
│   └── index.html
├── app.py                # Flask application
├── requirements.txt      # Project dependencies
└── setup.py             # Package configuration
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/MLPROJECT.git
cd MLPROJECT
```

2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model
```bash
python src/components/data_ingestion.py
```

2. Run the Flask application
```bash
python app.py
```

3. Access the web interface at `http://localhost:5000`

## Model Information

### Features Used
- Gender (categorical)
- Race/Ethnicity (categorical)
- Parental Level of Education (categorical)
- Lunch Type (categorical)
- Test Preparation Course (categorical)
- Reading Score (numerical)
- Writing Score (numerical)

### Model Performance
- R² Score: 0.88
- Algorithm: Random Forest Regressor
- Cross-validation: 5-fold

## API Reference

### Predict Endpoint
```http
POST /predict
```
Parameters:
| Parameter | Type | Description |
|-----------|------|-------------|
| gender | string | Student's gender |
| race_ethnicity | string | Student's race/ethnicity group |
| parental_level_of_education | string | Parent's education level |
| lunch | string | Lunch type (standard/free/reduced) |
| test_preparation_course | string | Test prep course status |
| reading_score | integer | Reading test score (0-100) |
| writing_score | integer | Writing test score (0-100) |

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Inspired by various educational performance prediction studies