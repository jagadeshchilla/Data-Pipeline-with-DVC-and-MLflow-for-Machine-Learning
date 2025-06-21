import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import mlflow
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/jagadeshchilla/machineLearningPipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "jagadeshchilla"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "7212e5a96e00f4974cee0cbb72f1a232a57faf50"

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,verbose=2,n_jobs=-1)
    grid_search.fit(X_train,y_train)
    return grid_search

## Load params from params.yaml
params = yaml.safe_load(open('params.yaml'))['train']
def train(data_path,model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X=data.drop(columns=['Outcome'])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/jagadeshchilla/machineLearningPipeline.mlflow")

    ## start the MLFLOW run
    with mlflow.start_run():
        # split the data into train and test
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=random_state)
        signature=infer_signature(X_train,y_train)

        ## define the hyperparameters grid

        param_grid={
            'n_estimators':[100,200],
            'max_depth':[5,10,None],
            'min_samples_split':[2,5],
            'min_samples_leaf':[1,2],
        }
        ## perform hyperparameter tuning
        grid_search=hyperparameter_tuning(X_train,y_train,param_grid)

        ## get the best model
        best_model=grid_search.best_estimator_

        ## predict and evaluate the model
        y_pred=best_model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(f"Accuracy: {accuracy}")


        ## log the additional metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimators",grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth",grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split",grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf",grid_search.best_params_['min_samples_leaf'])


        ## log the confusion matrix and classification report

        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)

        mlflow.log_text(str(cm),'confusion_matrix.txt')
        mlflow.log_text(cr,'classification_report.txt')

        # Skip model logging to avoid DagsHub compatibility issues
        # The model is still saved locally as a pickle file
        

        ## create the directory to save the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))


        print(f"Model saved to {filename}")

if __name__ == "__main__":
    train(params['data'],params['model_path'],params['random_state'],params['n_estimators'],params['max_depth'])
    
