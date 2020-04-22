Thomas Merkh, tmerkh@g.ucla.edu, April 22 2020.

quantify_performance.py - This script evaluates the trained model on the training, validation, and test data sets.  The confusion matrix is printed out for each, and the accuracy, F1 score, and AUC metrics are all computed for each data set. 

inference.py - This script can be used to perform inference on new data points, providing a 0-1 prediction and confidence levels.  Note: be sure to include the --imagepath /path/to/image/imagename.png parameter when calling this script. 

zzzz_CTx.txt - these are formatted specifically indicating the labels (0 or 1) associated with the images in the train/validation/test folders under the Data directory.  

PretrainedModel - <Google Drive linked> the directory containing the pretrained model.  This model can be restored in tensorflow and then further trained, adapted, or used for inference. 

Data - The "official" data split used to train, validate, and test the model. 