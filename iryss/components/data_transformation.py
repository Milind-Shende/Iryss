from iryss import utils
from iryss.entity import config_entity
from iryss.entity import artifact_entity
from iryss.logger import logging
from iryss.exception import IryssException
import os,sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from iryss.config import TARGET_COLUMN


class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTrasformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise IryssException(e, sys)
    

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:

        try:
            logging.info("Reading Train And Test File")
            train_df =pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(train_df.shape)
            logging.info(test_df.shape)

            numeric_features = ['AGE', 'HOSPID','NPR','NCHRONIC','ZIPINC_QRTL','DXn']
            logging.info(f"Numerical Column{numeric_features}")


            categorical_features = ['RACE','DRG','PAY1','PAY2','CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_DRUG','TRAN_IN','TRAN_OUT'] 
            logging.info(f"CategoricalColumn{categorical_features}")
            

            logging.info("Numerical and Categorical Pipeline Transformation")
            numeric_transformer= Pipeline(steps=[('scaler', MinMaxScaler(feature_range=(-1, 1)))])
            categorical_transformer = Pipeline(steps=[ ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))])

            logging.info("Numerical and Categorical Column Transformation")
            transformer = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                            ('cat', categorical_transformer, categorical_features)])

           

            logging.info("Spliting Train Into X_train And y_train")
           
            X_train = train_df.drop(TARGET_COLUMN,axis=1)
            y_train = train_df[TARGET_COLUMN]

            logging.info(X_train.shape)
            logging.info(y_train.shape)

            logging.info("Transforming X_train Column Transform")
            X_train=transformer.fit_transform(X_train)
            

            logging.info("Spliting Test Into X_test And y_test")
            
            X_test = test_df.drop(TARGET_COLUMN,axis=1)
            y_test = test_df[TARGET_COLUMN]

            logging.info(X_test.shape)
            logging.info(y_test.shape)

            logging.info("Transforming X_train Column Transform")
            X_test=transformer.transform(X_test)

                

            logging.info("train and test array concatenate")
            #train and test array
            train_arr = np.c_[X_train, y_train ]
            test_arr = np.c_[X_test, y_test]
            


            logging.info("Saved train and test array to save_numpy_array_data")
            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,obj=transformer)

            
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path= self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise IryssException(e, sys) 