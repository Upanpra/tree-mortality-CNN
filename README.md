# Enhancing Individual Tree Mortality Mapping: The Impact of Models, Data Modalities, and Classification Taxonomy
This is a code repository for our paper DOI (yet to come).

## Environment
To run our code, create a python virtual environment using the included `requirements.txt`. For training to run quickly, you will need a machine with a GPU.

## Training CNN model
To train our models use the included training scripts and data. You will need to unzip the data in the `data` folder and update the path to the dataset in the scripts.  

## Training ML model
To train our models use the included ipython notebook and data. You will need to unzip the data in the `data` folder and update the path to the dataset in the scripts. The uploaded script runs ml models for RGB you will need to change it to NAIP and Hyperspectral as needed.

## Running Inference
Our tranined checkpoints are avaiable inside data/model_checkpoints and an example of how to run the model on unlabeled data is shown in predict.py

## Data
The data for this project is available on Zenodo: <a href="https://zenodo.org/records/10114929" target="_blank">https://zenodo.org/records/10114929 </a>

## License
The code is released under the included MIT [license](LICENSE) while the data is released under a creative commons non-commercial license included with the data.
