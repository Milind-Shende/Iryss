from iryss.logger import logging
from iryss.exception import IryssException
from iryss.utils import get_collection_as_dataframe
import sys,os
from iryss.entity import config_entity
from iryss.components.data_ingestion import DataIngestion
from iryss.components.data_transformation import DataTransformation
# from iryss.components.model_trainer import ModelTrainer
# from iryss.components.model_evaluation import ModelEvaluation
# from iryss.components.model_pusher import ModelPusher


if __name__=="__main__":
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        #data transformation
        data_transformation_config = config_entity.DataTrasformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
        data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

    except Exception as e:
        raise IryssException(e, sys)