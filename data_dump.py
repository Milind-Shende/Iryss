import pymongo
import pandas as pd
import json
from typing import List

#Provide the MongoDB Localhost URL to Connect Python to MongoDB.

client= pymongo.MongoClient("mongodb+srv://Milind2487:mili%232487@milind2487.olvhy.mongodb.net/test")

DATA_FILE_PATH="D:\Project\Omdena_Iryss\sample_dataset_lung_prediction_cost.csv"
DATABASE_NAME="OmdenaIryss"
COLLECTION_NAME="Iryss"

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert Dataframe to json format to dump this records into mongodb
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #Insert Converted json Record To MongoDB Database
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)