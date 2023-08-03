# Image Plant Classification with CNN
Given a dataset of 1350 plant disease photos divided in 5 classes (270 images each):
* colpo_di_fuoco / fire_blight
* flavescenza / flavescence
* maculatura_bruna / brown spot
* peronospora / downy mildew
* ticchiolatura / scab
the aim of this project is to train the data (with different state of the art architectures and methods) and test the best models in order to be able to classify correctly the plant diseases.

The dataset used, was created mixing google images with photos taken on the field by agricultural technicians working in the italian plain of Pianura Padana.

The main work is available in the jupyter file: **image_plant_classification_Zanetti.ipynb**
The work is divided in 5 parts:
* Creating the Dataset and assigning the labels
* Deep model Preprocessing
* Building the network (LeNet implementation) and testing it
* AlexNet implementation
* Building a new model on top of features extracted (with the use of VGG16 model)

In the file **app_class_streamlit.py** a simple app for testing images was created by using the visual library of streamlit.
