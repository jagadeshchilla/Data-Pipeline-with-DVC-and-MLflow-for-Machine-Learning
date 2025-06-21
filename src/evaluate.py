import pandas as pd
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import mlflow
import os
import yaml
from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/jagadeshchilla/machineLearningPipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "jagadeshchilla"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "7212e5a96e00f4974cee0cbb72f1a232a57faf50"

params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=['Outcome'])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/jagadeshchilla/machineLearningPipeline.mlflow")


    ## load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    ## predict the model
    y_pred=model.predict(X)

    ## evaluate the model
    accuracy=accuracy_score(y,y_pred)
    print(f"Accuracy: {accuracy}")

    ## log the metrics
    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate(params['data'],params['model_path'])