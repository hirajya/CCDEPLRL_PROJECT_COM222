import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Dataset information
    dataset = "ibrahimfateen/wound-classification"
    
    # Create dataset directory if it doesn't exist
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    print(f"Dataset URL: https://www.kaggle.com/datasets/{dataset}")
    
    try:
        # Download the dataset
        api.dataset_download_files(dataset, path='dataset', unzip=True)
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()