**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./extra_test_images/bumpy-road.jpg "Traffic Sign 1"
[image3]: ./extra_test_images/children_crossing.jpg "Traffic Sign 2"
[image4]: ./extra_test_images/dangerous_curve_right.jpg "Traffic Sign 3"
[image5]: ./extra_test_images/general_caution.jpg "Traffic Sign 4"
[image6]: ./extra_test_images/go_straigh_or_left.jpg "Traffic Sign 5"
[image7]: ./extra_test_images/no_entry.jpg "Traffic Sign 6"
[image8]: ./extra_test_images/priority_road.jpg "Traffic Sign 7"
[image9]: ./extra_test_images/round_about_mandatory.jpg "Traffic Sign 8"
[image10]: ./extra_test_images/slippery_road.jpg "Traffic Sign 9"
[image11]: ./extra_test_images/speed_limit_30.jpg "Traffic Sign 10"
[image12]: ./extra_test_images/wild_animal_crossing.jpg "Traffic Sign 11"
[image13]: ./writeup_images/predictions1.png "Predictions1"
[image14]: ./writeup_images/predictions2.png "Predictions2"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ore92/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 123630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the class distribution of the training set

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 


I normalized the image data between 0 and 1 to improve convergence,keeping the color values because I belive color is a critical information in recognizing traffic signals; some signs are yellow, have red mixed with black etc


The orginal and normalized image look the same with matplotlib because it can handle displaying normalized images between 0 and 1


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| Convolution 1X1		| 1x1 stride,valid padding,outputs 28X28x28     |
| RELU					|												|
| Max pooling			| 2x2 stride 14x14x28							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x48 	|
| Convolution 1X1		| 1x1 stride,valid padding,outputs 10X10x72     |
| RELU					|												|
| Max pooling			| 2x2 stride 5x5x72							    |
| Flattening			| Output 1x1800					                |
| Fully connected		| Input:1x1800;Output 1x360    					|
| RELU					|												|
| Fully connected       | Input:1x360;Output 1x250   		            |
| RELU					|												|
| Fully connected/Output| Input:1x250;Output 1x43   		            |

 
 After each max pool and activation of fully connected layer, a dropout of 0.5 was applied as a regularization technique during trainig.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model on my CPU, with a target validation accuracy of about 95%. I achieved this with an EPOCH of 15, I used a batch size of 128(using all images to train) and a learning rate of 0.001. I used the AdamOptimizer to minimize the average softmax cross entropy.

#### 4. Describe the approach taken for finding a solution and getting 
I started using the LeNet Architecture with some changes. I increased the depth at each layer since there are more output classes(43) in this set than the original architure used for detecting numbers(10,0-9). I also added a 1x1 convolution at each layer to increase the numbers of parameters and as a really cheap way to make my network deeper. I used dropout as regularization technique at certain layer also. This model worked rather well and I did not have to tweak too many things

My final model results were:
* validation set accuracy of 0.965
* test set accuracy of 0.957
 

### Test a Model on New Images

#### 1. I chose 11 new images from the web to perform further tests on

Here are 11 German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12] 

from the images above, 9 out of 11 images were classified correcly (**81% accurate**);  two classified incorrectly one was a  speed limit signs that the numbers were wrong; the other was ahead only instead of ahead and left. Both predictions are reasonable guesses in context. Perharps a seperate classifier can be used to detect numbers on the speed limit signs.


Here are the results of the prediction and the certainty of predictions from softmax probablities:
![1][image13]
![2][image14]


The model was able to correctly guess 9 of the 11 traffic signs, which gives an accuracy of 81%. This compares favorably to the accuracy on the test set of 95.7% especially since it's guesses were not far off.