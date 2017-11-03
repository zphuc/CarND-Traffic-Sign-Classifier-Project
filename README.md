#**Traffic Sign Recognition**
---

**Project to Build a Traffic Sign Recognition**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: images/Bchar.png "Visualization"
[image2]: images/origImg.png "Original"
[image3]: images/grayImg.png "Grayscaling and normalized"
[image15]: newImages/01.jpg "Traffic Sign 1"
[image13]: newImages/11.jpg "Traffic Sign 2"
[image16]: newImages/14.jpg "Traffic Sign 3"
[image12]: newImages/22.jpg "Traffic Sign 4"
[image14]: newImages/25.jpg "Traffic Sign 5"
[image11]: newImages/40.jpg "Traffic Sign 6"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
**Load the data set**

I used the notebook template which was provided to load the data  
You're reading it! and here is a link to my [project code](https://github.com/zphuc/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

**Data Set Summary & Exploration**

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the traning data set.   
It is a bar chart showing the samples per class

![alt text][image1]

**Design and Test a Model Architecture**

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Referring the paper of [Sermanet and LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), i can find that,
  * Using the grayscale images can reached more accuracy than the color ones (98.97% -> 99.17%)
  * Non-normalized color channels may yielded overall worse performance of model.

Therefore, I decided to convert the color images to the grayscale images in the first step, then normalized the grayscale images.


In here, in order to convert the color images to grayscale, I did use the method (R+G+B)/3 which was given in the first classroom of the CNN lesson. This method  was easy for coding.  

In addition, I normalized the grayscale image data using the suggested normalization method (pixel - 128)/ 128. This method can normalize the image data to the range (-1,1), that makes good sense in my experience.

I was also thinking to consider carefully the steps again with other grayscale and normalization methods if the below model could not reach the required accuracy.

Here is an example of some traffic sign images before and after grayscaling and normalized.

![alt text][image2]
![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the the [LeNet](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb) architecture referring from the lecture.  
However, the number of our classes (43) is bigger than the final outputs (10) of the given LeNet example.  
In here, I has modified the output depths of the layers by multiplying **dd** factor to investigate the effect of depths.  
The final output was set to be 43, instead of 10 as shown in LeNet example.  

By changing the **dd** factor of 1, 2, 4, I found that it could increase the accuracy.

Here is my final model consisted of the following layers (**dd** = 4 ~ 43/10).

| Layer         		|     Description	        					|
|:-----------------:|:---------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 3x3   | 1x1 stride, same padding, outputs 28x28x(6*dd) 	|
| RELU					    |						            						|
| Max pooling	      | 2x2 stride,  outputs 14x14x(6*dd)	|
| Convolution 3x3   | 1x1 stride, same padding, outputs 10x10x(16*dd) 	|
| RELU					    |								            				|
| Max pooling	      | 2x2 stride,  outputs 5x5x(16*dd) 	|
| Flatten		        | outputs 400*dd      						  |
| Fully connected		| outputs 120*dd     								|
| RELU					    |												            |
| Fully connected		| outputs 84*dd      								|
| RELU				    	|												            |
| Fully connected		| outputs 43        								|




####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I have done the following steps to optimize the hyperparameters

* Step 1: Consider the changing of **d** factor
   * **dd**        = 1, 2, 4
   * batch size    = 128
   * epochs        = 10
   * learning rate = 0.001  

  In here, the accuracies of the validation dataset were 0.905, 0.921 and 0.948, respectively. The accuracy increased with the increasing of **d**.   
  Then I fixed the **d** factor of 4 and went to the next step

* Step 2: Consider the changing of batch size
  * **dd**        = 4
  * batch size    = 64, 128, 256
  * epochs        = 10
  * learning rate = 0.001  

  In here, the accuracies of the validation dataset were 0.955, 0.950 and 0.929, respectively. Small batch size showed better results and the batch size of 128 is seemly enough. Therefore I fixed the batch size of 128 and went to the following step.

* Step 3: Consider the changing of epochs
  * **dd**        = 4
  * batch size    = 128
  * epochs        = 10, 20 ,40
  * learning rate = 0.001  

  In here, the accuracies of the validation dataset were 0.950, 0.956 and 0.961, respectively. Large epochs showed better results but did with much more computation time. However, the result of epochs of 10 is quite good, hence, I used the value and went to the next step.

* Step 4: Consider the changing of learning rate
  * **dd**        = 4
  * batch size    = 128
  * epochs        = 10
  * learning rate = 0.001, 0.005, 0.01  

  In here, the accuracies of the validation dataset were 0.950, 0.922 and 0.054, respectively. Large learning rate showed bad result. Hence, I decided to use the learning rate of 0.001 to train the model.

???????????????????????????????????????????????????????????????????????  
Here are the final hyperparameters that I used to train the model
  * **dd**        = 4
  * batch size    = 128
  * epochs        = 10
  * learning rate = 0.001

My final model results were as follows (see my [Ipython](Traffic_Sign_Classifier.ipynb) notebook)
  * training set accuracy of 0.999
  * validation set accuracy of 0.952
  * test set accuracy of 0.933  

???????????????????????????????????????????????????????????????????????

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I have chosen the iterative approach
* What was the first architecture that was tried and why was it chosen?  
>I used the original [LeNet](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb) architecture which was introduced and suggested at the classroom as the first architecture.

* What were some problems with the initial architecture?  
>For the initial architecture with default parameters, and only change the final outputs of 10 to 43 to obtain the same number of sign classes, the accuracies of the validation dataset were about 0.90 or less. I could not reach to the required accuracy of 0.930.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  

  >It is well-known that depths of the layers are important in LeNet model. Moreover, I want to keep the structure of the original LeNet example, I decide only to change the depths of layers by the multiply factor **dd** (it is just my ideal).  
  After investigate the changing of **dd** factor, I found that it worked well and increase effectively the accuracy (See the result of the step 1 in previous answer).

* Which parameters were tuned? How were they adjusted and why?
  >the **d** factor, batch size, epochs and learning rate were the parameters that I have tuned. Please see the previous answer to know how they were adjusted.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
 > I was also thinking to increase more the convolution layer that might get more accuracy, if the **dd** factor did not work well. It would be my further work.


**Test a Model on New Images**

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image11] ![alt text][image12] ![alt text][image13]
![alt text][image14] ![alt text][image15]

I think that the below images might be difficult to classify because,  
* 1st image: the sign color is quite the same with the sky background
* 3nd image: the sign is in inclined
* 5th image: the sign is quite small with much background in comparing with training dataset

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

However, here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Roundabout mandatory      		        | Roundabout mandatory   									|
| Bumpy road     			                  | Bumpy road 										          |
| Right-of-way at the next intersection	| Right-of-way at the next intersection		|
| Road work	      		                  | Road work					 				              |
| Speed-limit(30km/h)			              | Speed-limit(30km/h)      		  					|


The model was work well for the 5 traffic signs, which gave an accuracy of about 100%. This compares favorably to the accuracy on the test set of training data


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the [Ipython](Traffic_Sign_Classifier.ipynb) notebook.

For the 5th image, the model gave the lowest probability of 0.991 among images, and it is also interesting to see the top five soft max probabilities as follows. All of them are the speed-limit signs with different speed values

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .991       			      | Speed-limit(30km/h)    									|
| .006     				      | Speed-limit(80km/h)  										|
| .000					        | Speed-limit(70km/h) 											|
| .000	      		      | Speed-limit(100km/h) 					 				|
| .000				          | Speed-limit(20km/h)       							|
