%%bash
pip install kaggle

%%bash
mkdir /home/ec2-user/.kaggle
mv /home/ec2-user/SageMaker/kaggle.json /home/ec2-user/.kaggle
chmod 600 /home/ec2-user/.kaggle/kaggle.json
%%bash
ls -al /home/ec2-user/.kaggle/


%%bash
kaggle datasets download --unzip paultimothymooney/breast-histopathology-images

%%bash
rm -rf IDC_regular_ps50_idx5

%%bash
mkdir images/0
mkdir images/1

import os
for path, subdirs, files in os.walk('images'):
    for name in files:
        filename = os.path.join(path,name)
        if name.endswith('class0.png'):
            destination_class = '0'
        else:
            destination_class = '1'
        os.rename(filename,os.path.join('images',destination_class,name))

%%bash
cd images/0
ls -l|wc -l

%%bash
cd images/1
ls -l|wc -l

%%bash
shopt -s extglob
cd images
rm -rf !("0"|"1")


%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('images/0/16551_idx5_x2651_y2001_class0.png')
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('images/1/15516_idx5_x2101_y1751_class1.png')
imgplot = plt.imshow(img)
plt.show()

%%bash
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py
chmod +x im2rec.py

%%bash

python im2rec.py --list --recursive --test-ratio 0.3 --train-ratio 0.7 images images/
%%bash

python im2rec.py --num-thread 4 --pass-through images_train.lst images
python im2rec.py --num-thread 4 --pass-through images_test.lst images

%%bash

aws s3 cp images_train.rec s3://sagemaker-gwu-capstone-2019/breast-cancer-detection/input/recordio/train/

%%bash
aws s3 cp images_test.rec s3://sagemaker-gwu-capstone-2019/breast-cancer-detection/input/recordio/test/

import time
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.model import Model
from sagemaker.predictor import RealTimePredictor
import json
import numpy as np

#Number of output classes
num_classes = 2

# Number of training samples in the training set # number obtained from train.lst
num_training_samples = 194266

# Number of layers for underlying neural network
num_layers = 18

# Batch size for training  # How many images are provided to the model for training
mini_batch_size = 128

#Input image shape for the training data
image_shape = '3,50,50'

# Augmentation type #i have an unbalanced dataset 
augmentation_type = 'crop_color_transform'

#Number of epoch
epochs = 5

# Learning Rate
learning_rate =0.01

# Enable Transfer Learning
use_pretrained_model = 1


job_name_prefix = 'breast-cancer-detection'
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
job_name = job_name_prefix + timestamp


bucket = 'sagemaker-gwu-capstone-2019'
input_prefix = 'breast-cancer-detection/input/recordio'
input_train = 's3://{}/{}/train/'.format(bucket, input_prefix)
input_test = 's3://{}/{}/test/'.format(bucket, input_prefix)

output_prefix = 'breast-cancer-detection/output'
output_path = 's3://{}/{}/'.format(bucket, output_prefix)


instance_count = 1
instance_type = 'ml.p2.xlarge'
volume_size_gb = 50


role = get_execution_role()
training_image = get_image_uri(boto3.Session().region_name, 'image-classification')

train_timeout = 360000
sagemaker_session = sagemaker.Session()
estimator = sagemaker.estimator.Estimator(training_image, 
                                          role, 
                                          train_instance_count=instance_count,
                                          train_instance_type=instance_type,
                                          train_volume_size=volume_size_gb,
                                          train_max_run=train_timeout,
                                          output_path=output_path, 
                                          sagemaker_session=sagemaker_session,
                                          input_mode='Pipe')

estimator.set_hyperparameters(num_classes=num_classes,
                              num_training_samples=num_training_samples,
                              num_layers=num_layers,
                              mini_batch_size=mini_batch_size,
                              image_shape=image_shape,
                              augmentation_type=augmentation_type,
                              epochs=epochs,
                              learning_rate=learning_rate,
                              use_pretrained_model=use_pretrained_model)
                              


import time

import boto3

import sagemaker

from sagemaker import get_execution_role

from sagemaker.amazon.amazon_estimator import get_image_uri

from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

from sagemaker.model import Model

from sagemaker.predictor import RealTimePredictor

import json

import numpy as np

Configure Built In Image Classification Algorithm

Configure Hyperparameters

#Number of output classes

num_classes = 2

​

# Number of training samples in the training set # number obtained from train.lst

num_training_samples = 194266

​

# Number of layers for underlying neural network

num_layers = 18

​

# Batch size for training  # How many images are provided to the model for training

mini_batch_size = 128

​

#Input image shape for the training data

image_shape = '3,50,50'

​

# Augmentation type #i have an unbalanced dataset 

augmentation_type = 'crop_color_transform'

​

#Number of epoch

epochs = 5

​

# Learning Rate

learning_rate =0.01

​

# Enable Transfer Learning

use_pretrained_model = 1

Create a Unique Job Name

job_name_prefix = 'breast-cancer-detection'

timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())

job_name = job_name_prefix + timestamp

Specify the input path for the job

bucket = 'sagemaker-gwu-capstone-2019'

input_prefix = 'breast-cancer-detection/input/recordio'

input_train = 's3://{}/{}/train/'.format(bucket, input_prefix)

input_test = 's3://{}/{}/test/'.format(bucket, input_prefix)

Specify the output path for the job

output_prefix = 'breast-cancer-detection/output'

output_path = 's3://{}/{}/'.format(bucket, output_prefix)

Configure training instances

instance_count = 1

instance_type = 'ml.p2.xlarge'

volume_size_gb = 50

​

Execution role and training image URI for Image Classification

role = get_execution_role()

training_image = get_image_uri(boto3.Session().region_name, 'image-classification')

Configure train timeout

train_timeout = 360000

create a sagemaker estimator

sagemaker_session = sagemaker.Session()

estimator = sagemaker.estimator.Estimator(training_image, 

                                          role, 

                                          train_instance_count=instance_count,

                                          train_instance_type=instance_type,

                                          train_volume_size=volume_size_gb,

                                          train_max_run=train_timeout,

                                          output_path=output_path, 

                                          sagemaker_session=sagemaker_session,

                                          input_mode='Pipe')

estimator.set_hyperparameters(num_classes=num_classes,

                              num_training_samples=num_training_samples,

                              num_layers=num_layers,

                              mini_batch_size=mini_batch_size,

                              image_shape=image_shape,

                              augmentation_type=augmentation_type,

                              epochs=epochs,

                              learning_rate=learning_rate,

                              use_pretrained_model=use_pretrained_model)

Create a training job

s3_input_train = sagemaker.s3_input(s3_data=input_train, content_type='application/x-recordio')

s3_input_validation = sagemaker.s3_input(s3_data=input_test, content_type='application/x-recordio')
estimator.fit({
    'train': s3_input_train,
    'validation': s3_input_validation
}, job_name=job_name)
hyperparameter_ranges = {
    'learning_rate': ContinuousParameter(0.001, 1.0),
    'mini_batch_size': IntegerParameter(64, 128),
    'optimizer': CategoricalParameter(['sgd', 'adam'])
}

objective_metric_name = 'validation:accuracy'
objective_type='Maximize'
max_jobs=2
max_parallel_jobs=2

job_name_prefix = 'bcd-tuning'
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
job_name = job_name_prefix + timestamp

tuner = HyperparameterTuner(estimator=estimator, 
                            objective_metric_name=objective_metric_name, 
                            hyperparameter_ranges=hyperparameter_ranges,
                            objective_type=objective_type, 
                            max_jobs=max_jobs, 
                            max_parallel_jobs=max_parallel_jobs)
                            
tuner.fit({
    'train': s3_input_train,
    'validation': s3_input_validation
}, job_name=job_name)
tuner.wait()

role = get_execution_role()
hosting_image = get_image_uri(boto3.Session().region_name, 'image-classification')
instance_count = 1
instance_type = 'ml.m4.xlarge'
model_name_prefix = 'bcd-image-sdk'
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
model_name = model_name_prefix + timestamp
model_artifacts_s3_path = 's3://sagemaker-gwu-capstone-2019/breast-cancer-detection/output/bcd-tuning-2019-11-12-13-27-23-002-c75e5348/output/model.tar.gz'
model = Model(
    name=model_name,
    model_data=model_artifacts_s3_path,
    image=hosting_image,
    role=role,
    predictor_cls=lambda endpoint_name, sagemaker_session: RealTimePredictor(endpoint_name, sagemaker_session)
)
endpoint_name_prefix = 'breast-cancer-detection-sdk-ep'
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
endpoint_name = endpoint_name_prefix + timestamp

predictor = model.deploy(
    endpoint_name=endpoint_name,
    initial_instance_count=instance_count,
    instance_type=instance_type
)

def predict_breast_cancer(image_path):
    with open(image_path, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)
    response = predictor.predict(payload)
    result = json.loads(response)
    print('Probabilities for all classes: ', result)
    predicted_class = np.argmax(result)
    if predicted_class == 0:
        print('Breast cancer not detected')
    else:
        print('Breast cancer detected')
predict_breast_cancer('images/0/10275_idx5_x351_y851_class0.png')
predict_breast_cancer('images/1/10275_idx5_x951_y751_class1.png')
sagemaker.Session().delete_endpoint(predictor.endpoint)
