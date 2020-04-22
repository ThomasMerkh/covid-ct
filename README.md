# covid-ct
<center><b>About</b></center>
This repository contains a deep convolutional network trained on CT data for the binary classification of COVID/Non-COVID.  <i>Transfer learning</i> was used here, where I utilized a pre-trained COVID-Net model (see https://arxiv.org/abs/2003.09871v1), and fine-tuned the parameters of the network using the training set.  After selecting the best performing model on the validation set, the performance was quantified using three metrics: accuracy, F1-score, and the area under the ROC curve.

<center><b>Requirements</b></center>
*To be updated 

<center><b>Usage</b></center>

<ol>
  <li>Clone this repository on your local device</li>
  <li>Unzip the training and validation data sets, keep them in their respective directories </li>
  <li>Download and unzip the pre-trained model (approx 2Gbs) here: https://drive.google.com/open?id=1MYRGcs7aMpzbDEuhpvZshdwaOFq8ZKvF</li>
  <li>Run the script quantify_performance.py and/or inference.py</li>
</ol>

*Tips
<ul>
  <li>Do not rearrange the sub-directories or rename them</li>
  <li>To run inference.py, one must include a --imagepath argument specifying the path to an image file one wishes to perform inference on</li>
  <li>Besides the previous point, the default arguments for quantify_performance.py and inference.py should be set so one may run this model directly without adjustment</li>
 </ul>
 
 <center><b>Results</b></center>
 *To be updated
