#%% By Janis@250101
from pymongo import MongoClient
import pandas as pd
def get_mongo(windcode:str):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['stock_data']
    collection = db['minute_data_15min']
    query = {"windcode": windcode}
    documents = collection.find(query)
    df = pd.DataFrame(list(documents))
    if '_id' in df.columns:
        del df['_id']
    return df

from WindPy import w
from pymongo import MongoClient
import pandas as pd

def fetch_wind_data(windcode: str, start_date: str, end_date: str):
    w.start()
    data = w.wsd(windcode, "open,high,low,close,volume", start_date, end_date, "Fill=Previous")
    df = pd.DataFrame(data.Data, index=data.Fields, columns=data.Times).T
    df['windcode'] = windcode
    return df

def store_to_mongo(df: pd.DataFrame):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['stock_data']
    collection = db['minute_data_15min']
    collection.insert_many(df.to_dict('records'))

def get_mongo(windcode: str):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['stock_data']
    collection = db['minute_data_15min']
    query = {"windcode": windcode}
    documents = collection.find(query)
    df = pd.DataFrame(list(documents))
    if '_id' in df.columns:
        del df['_id']
    return df

# Example usage
windcode = "000001.SZ"
start_date = "2023-01-01"
end_date = "2023-12-31"
df = fetch_wind_data(windcode, start_date, end_date)
store_to_mongo(df)
retrieved_df = get_mongo(windcode)
print(retrieved_df)
