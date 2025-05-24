import os
import shutil
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
        
        # Keep only Bruises and Normal folders
        dataset_path = os.path.join('dataset', 'Wound_dataset copy')
        if os.path.exists(dataset_path):
            # List all folders
            folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
            
            # Remove unwanted folders
            for folder in folders:
                if folder not in ['Bruises', 'Normal']:
                    folder_path = os.path.join(dataset_path, folder)
                    shutil.rmtree(folder_path)
                    print(f"Removed folder: {folder}")
            
            print("Cleaned up dataset to keep only Bruises and Normal folders")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()