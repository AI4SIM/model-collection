#!/bin/sh

USE_CASE=$1

if [ "$USE_CASE" == combustion_gnn ] 
then
    USE_CASE_PATH = model-collection/combustion/gnns
elif [ "$USE_CASE" == combustion_unet ] 
then
    USE_CASE_PATH = model-collection/combustion/unets
elif [ "$USE_CASE" == wf_gwd ]
then
    USE_CASE_PATH = model-collection/weather_forecast/gwd
else
    echo "use-case not implemented !"
fi

# install requirements
python -m pip install -r $USE_CASE_PATH/requirements.txt \
    -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
    --extra-index-url https://download.pytorch.org/whl/cu113