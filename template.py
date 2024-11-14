import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "movierecomender"

list_of_files = [
    # Core project structure
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_transformation.py",  # Data ingestion script
    f"src/{project_name}/components/feature_extraction.py",  # Image feature extraction script
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/data_merging.py",
    f"notebook/eda.ipynb",
    f"notebook/handled_data.ipynb",
    f"app/app.py",
    f"app/index.html",
    f"app/recomendation.html",
    f"templates/index.html",
    f"templates/recomendation.html",
    f"templates/404.html",
    f"templates/505.html",


  

    # Unit test structure
    f"tests/test_data_transformation.py",
    f"tests/test_feature_extraction.py",
    f"tests/test_model_training.py",
    
  # DVC structure
    "dvc.yaml",  
    ".dvc/config", 
    ".dvcignore", 
    "requirements.txt", 
    "setup.py", 
    "loging.py",
    "custom_exception.py",
    ".gitignore"



]

# Create the directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as file:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
