import os
from roboflow import Roboflow

#export ROBOFLOW_API_KEY='your_api_key'
api_key = os.environ.get("ROBOFLOW_API_KEY")

# Inserisci la tua API key personale
rf = Roboflow(api_key=api_key)

# Accedi al workspace e al progetto
project = rf.workspace("ticon").project("dogs-and-cats-t43j2")
version = project.version(1)

# Scarica il dataset pronto per YOLOv8
dataset = version.download("yolov8")