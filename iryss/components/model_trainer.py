from iryss.logger import logging
from iryss.exception import IryssException
from iryss import utils
from iryss.entity import config_entity
from iryss.entity import artifact_entity
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
import os,sys
import numpy as np



class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,data_transformation_artifact:artifact_entity.DataTransformationArtifact):

        try:
            logging.info(f"{'>>'*20} Model Trainer{'<<'*20}")
            self.model_trainer_config =model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise IryssException(e,sys)

    def train_model(self,X,y):
        try:
            logging.info("Training Model With XgboostRegression")
            xgb_regressor=xg.XGBRegressor(eval_metric='rmsle',objective ='reg:linear',
                  n_estimators = 10, seed = 123)
            xgb_regressor.fit(X,y)
            return xgb_regressor
        except Exception as e:
            raise IryssException(e,sys)
    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            X_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Train the model")
            model=self.train_model(X=X_train,y=y_train)

            # logging.info(f"Calculating RMSE train score")
            yhat_train=model.predict(X_train)
            rmse_train_score=np.sqrt(MSE(y_true=y_train,y_pred=yhat_train,squared=False))
           
            train_score=model.score(X_train, y_train)
            # logging.info(f"Calculating train score{train_score}")
            # logging.info(f"Calculating RMSE test score")
            yhat_test=model.predict(X_test)
            rmse_test_score=np.sqrt(MSE(y_true=y_test,y_pred=yhat_test,squared=False))
            
            test_score=model.score(X_test, y_test)
            # logging.info(f"Calculating test score{test_score}")
            logging.info(f"train score:{train_score} and tests score {test_score}")
            #Check for overfitting or underfitting or expected score
            """Overfitting mean good accuracy on training score but not getting good accuracy on test score

               Underfitting means we are not getting good accuracy on both train and test accuracy

               Expected score means which decided by us.
            
            """
            logging.info(f"Checking if our model is underfitting or not")
            if rmse_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {test_score}")

            logging.info(f"Checking if model is overfitting or not")
            diff=abs(train_score-test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test Score Difference:{diff} is more than overfitting threshold{self.model_trainer_config.overfitting_threshold}")


            logging.info(f"Saving model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)


            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,train_score=train_score,test_score=test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise IryssException(e,sys)