# droneface

# init docker
docker run --gpus all -it -v path/to/folder/code:path/to/folder/code --name deepo-dl-cv-cuda11 pluto/deepo-dl-opencv-cuda11.0 bash

# install some pip packets
cd path/to/folder/code
pip install -r requirements.txt

# demo
python test.py