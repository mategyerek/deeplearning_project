#Deeplearning project
To recreate environment run conda env create -f environment.yml

##Training
Run train.py to trin the model. Set load to True to further train an existing model. The gridsearch can be performed by inseting the commented lines above line 91 in the for loop. The training generatesd the training and performance curves shown in the report. The model and the training curves will be named according to the hyperparameters used.
##Inference
Run inference.py to do predictions on the test set. This will generate the examples png. Adjust the batch size to see a different number of images. Change the filename in the load to use a different model.