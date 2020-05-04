# covid-ct
<br>
<center><b>About</b></center>
<br>
This repository contains a deep convolutional network trained on CT data for the binary classification of COVID/Non-COVID.  <i>Transfer learning</i> was used here, where I utilized a pre-trained COVID-Net model (see https://arxiv.org/abs/2003.09871v1), and fine-tuned the parameters of the network using the training set.  After selecting the best performing model on the validation set, the performance was quantified using three metrics: accuracy, F1-score, and the area under the ROC curve.

<br>
<center><b>Requirements</b></center>
<br>

This code was run on a linux device equipped with the following packages:
<ul>
  <li>Python               3.7.5</li>
  <li>numpy                1.18.2</li>  
  <li>opencv-python        4.2.0.34</li>       
  <li>scikit-learn         0.22.2.post1</li>
  <li>scipy                1.4.1</li>
  <li>tensorflow           1.15.0</li>  
</ul>

I expect that any collection of recently updated Python modules + Tensorflow 1.15 will be able to run these scripts without issue.

<br>
<center><b>Usage</b></center>
<br>
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
 
 <br>
 <center><b>Results</b></center>
 <br>
The model's performance on three metrics of interest here are as follows.

<ul>
<li>Prediction Accuracy:  Non-COVID - 99%, COVID - 93.9%</li>
<li>The F1-Score: 0.9674 </li>
<li>The AUC: 0.9646 </li>
</ul>                        	
<br>

<br> 
For any questions or concerns:
*tmerkh@g.ucla.edu

Last, code to continue training the model listed here is throwing some weird tensrflow error I have yet to figure out.  It will be fixed and uploaded as soon as possible. 
