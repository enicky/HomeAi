#from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import logging
from colorama import init, Fore, Back
from utils.logging import ColorLogger
import glob
import pandas as pd


init(autoreset=True)
logging.setLoggerClass(ColorLogger)
logger = logging.getLogger(__name__)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARN)

class BlobRelatedClass:
    def start_downloading_data(self):
        logger.info("Start processing files from Azure to Local")
        self.clean_data_folder()
        
        account_url = "https://dapraidata.blob.core.windows.net"
        #credential = DefaultAzureCredential()
        credential = os.getenv('blob_credential')
        

        with BlobServiceClient(account_url, credential=credential) as blob_service_client:
            self.list_blobs_flat(blob_service_client, "sensordata")
            logger.debug('Start merging all csv files into 1')
            self.merge_csv_files(target_filename='merged.csv', base_path='data/')
            logger.debug('Finished merging and sorting csv files')
            self.clean_data_folder(False)
            
            
    def clean_data_folder(self, include_merged=True):
        folder = os.path.join( 'data')
        folderExists = os.path.exists(folder)
        if not folderExists:
            logger.info('Folder didnt exist. So Create it')
            os.makedirs(folder)
            
        if folderExists:
            logger.info(f'Start deleting file in folder python/data')
            for filename in os.listdir(folder):
                if not include_merged and "merged" in filename:
                    continue
                logger.debug(f'[] Delete file {filename}')
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        logger.info('Finished clearing data folder.')
        
    def download_blob_to_file(self, blob_service_client: BlobServiceClient, container_name, blob_name):
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        with open(file=os.path.join(r'data', blob_name), mode="wb") as b:
            download_stream = blob_client.download_blob()
            b.write(download_stream.readall())
            
    def list_blobs_flat(self, blob_service_client: BlobServiceClient, container_name):
        
        container_client = blob_service_client.get_container_client(container=container_name)

        blob_list = container_client.list_blobs()

        for blob in blob_list:
            logger.info(f"Download : {blob.name}")
            self.download_blob_to_file(blob_service_client=blob_service_client, container_name=container_name,
                                       blob_name=blob.name)
        logger.info(f'Finished all blobs')
    def merge_csv_files(self, target_filename: str, base_path: str = 'python/data') -> None:
        
        # Use glob to find all CSV files in the directory
        csv_files = glob.glob(f'{base_path}/export-*.csv')

        # Read and concatenate all CSV files
        df_list = [pd.read_csv(file) for file in csv_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        print('-------')

        # Convert the date column to datetime
        merged_df['date'] = pd.to_datetime(merged_df['Time'])
        print(f'merged ')
        print('------')

        # Sort the DataFrame by date
        sorted_df = merged_df.sort_values(by='date')
        print('sorted')        
        print('-------')

        filename = f'{base_path}/merged_and_sorted_file.csv'
        if os.path.exists(filename):
            os.remove(filename)
        # Save the sorted DataFrame to a new CSV file
        sorted_df.to_csv(filename, index=False)
        

def main():
    
    logger.info("Start processing files from Azure to Local")
    b = BlobRelatedClass()
    b.clean_data_folder()
    
    account_url = "https://dapraidata.blob.core.windows.net"
    #credential = DefaultAzureCredential()
    credential = os.getenv('blob_credential')
    

    with BlobServiceClient(account_url, credential=credential) as blob_service_client:
        b.list_blobs_flat(blob_service_client, "sensordata")
        logger.debug('Start merging all csv files into 1')
        b.merge_csv_files(target_filename='merged.csv', base_path='data/')
        logger.debug('Finished merging and sorting csv files')
        b.clean_data_folder(False)
        

if __name__ == "__main__":
    main()
    
    
