import os
import tarfile
import logging
import subprocess
import shutil
import json
import zipfile

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# dataset info
DATASET_NAME = "catwhisker/floorplancad-dataset"
FILES_TO_DOWNLOAD = [
    "test-00.tar.xz",
    "train-00.tar.xz",
    "train-01.tar.xz",
]

# define directories
download_directory = "./downloaded_data/"
extract_directory = "./floorplancad-dataset/"

# create directories
os.makedirs(download_directory, exist_ok=True)
os.makedirs(extract_directory, exist_ok=True)

# setup kaggle api
def setup_kaggle_credentials():
    # get env credentials
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        logger.error("Kaggle credentials (KAGGLE_USERNAME, KAGGLE_KEY) not found in environment variables.")
        raise ValueError("Kaggle credentials must be set as environment variables.")

    kaggle_creds = {
        "username": username,
        "key": key
    }
    
    # create .kaggle dir
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # write credentials
    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
        json.dump(kaggle_creds, f)
    
    # set permissions
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
    
    logger.info("Kaggle API credentials set up successfully from environment variables")

# download kaggle dataset
def download_kaggle_dataset():
    logger.info(f"Downloading dataset {DATASET_NAME}")
    
    try:
        setup_kaggle_credentials()
            
        # download specific files
        for file_name in FILES_TO_DOWNLOAD:
            # kaggle api adds .zip
            zip_file_path = os.path.join(download_directory, file_name + ".zip")
            tar_file_path = os.path.join(download_directory, file_name)
            
            # skip if exists
            if os.path.exists(tar_file_path):
                logger.info(f"File {file_name} already exists, skipping download")
                continue
                
            logger.info(f"Downloading {file_name}")
            cmd = f"kaggle datasets download -d {DATASET_NAME} -f {file_name} -p {download_directory} --force"
            subprocess.run(cmd, shell=True, check=True)
            
            # extract .zip
            if os.path.exists(zip_file_path):
                logger.info(f"Extracting zip file: {zip_file_path}")
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(download_directory)
                
                # remove .zip
                os.remove(zip_file_path)
                logger.info(f"Removed zip file: {zip_file_path}")
            
            logger.info(f"Successfully downloaded {file_name}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False

# extract tar.xz files
def extract_files():
    for file_name in FILES_TO_DOWNLOAD:
        file_path = os.path.join(download_directory, file_name)
        
        try:
            # check file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue
                
            # get base name
            base_name = os.path.basename(file_path).split('.')[0]
            target_dir = os.path.join(extract_directory, base_name)
            
            # create sub-directory
            os.makedirs(target_dir, exist_ok=True)
            
            logger.info(f"Extracting {file_path} to {target_dir}")
            
            # extract archive
            with tarfile.open(file_path) as tar:
                # filter svg files
                svg_members = [m for m in tar.getmembers() if m.name.lower().endswith('.svg')]
                tar.extractall(path=target_dir, members=svg_members)
                
            logger.info(f"Successfully extracted {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting {file_path}: {str(e)}")

# main
if __name__ == "__main__":
    # download files
    if download_kaggle_dataset():
        # extract files
        extract_files()
        
        # print summary
        file_count = 0
        for root, _, files in os.walk(extract_directory):
            file_count += len(files)
        
        logger.info(f"Total files extracted: {file_count}")
    else:
        logger.error("Failed to download dataset files. Extraction aborted.")