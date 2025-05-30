from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    # Step 1: Ingest data
    df = ingest_df(data_path)
    # Step 2: Clean and split data
    X_train, X_test, y_train, y_test = clean_df(df)
    # Step 3: Train model
    model = train_model(X_train, y_train)
    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    # Update the path as needed
    train_pipeline(data_path="data/StudentsPerformance.csv")()
    # To run the pipeline, use: python src/pipeline/train_pipeline.py
    # Make sure ZenML is installed and initialized: pip install zenml

