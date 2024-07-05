from zenml import step
import pandas as pd
import logging

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    try:
        data_ingest = IngestData(data_path=data_path)
        df = data_ingest.get_data()
        return df
    except Exception as e:
        logging.info(e)
        raise e
