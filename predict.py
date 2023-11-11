
from src.model import MortalityCNN
from src.predict.dataset import get_predict_loader
from src.predict.predictor import predictCNN

from src.functions import *
from src.data_loader import transforms_aug
import time


# setting up train and test folder
train_folder = "data/training_data/4classes/rgb/train_rgb" # Assumes you have downloaded our data and placed it at the repo root

mean, std = get_mean_std(train_folder)
print(mean)
print(std)
print("RGB MCNN model")

# set all the below parameters change based on which data type cnn is being running for
INPUT_CHANNELS = 4
INPUT_SIZE = [128, 128]
NUM_CLASSES = 4            # The number of output classes. In this case, from 1 to 4
NUM_EPOCHS = 100           # change to 200 # The number of times we loop over the whole dataset during training
BATCH_SIZE = 16            # Change to 16 or 32 The size of input data took for one iteration of an epoch
LEARNING_RATE = 1e-3       # The speed of convergence


# prediction for the inference area
tiff_input_folder = "data/2018/unlabelled/" # directory of your unlabeled crops
tiff_output_folder = "data/prediction/2018" # directory of predicted files, with the same basename as each input file and the predicted class index appended to the end

# Define model name
model_name = "MortalityCNN"

model_path = "data/models/checkpoint/ "#path to saved model checkpoint

# use transforms_aug function from data loader to do augmentation
transform_train, transform_test = transforms_aug(INPUT_SIZE, mean, std)

predict_loader = get_predict_loader(tiff_input_folder, INPUT_SIZE, mean, std, BATCH_SIZE)


net = MortalityCNN(input_size=[INPUT_CHANNELS] + INPUT_SIZE)

# calling cnn functions either resnet or droughtCNN

net = load_model(net, model_path)
print(net)

start = time.time()
print(start/60)

predictCNN(net, predict_loader, tiff_output_folder)

end = time.time()
print(end)
print(f" Elapsed time: {(end-start)/60} minutes")
print(f" Elapsed time: {(end-start)/60/60} hours")
print(time.time())
