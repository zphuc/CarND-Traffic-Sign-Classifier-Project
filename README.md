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

[image1]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/images/Bchar.png "Visualization"
[image2]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/images/origImg.png "Original"
[image3]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/images/grayImg.png "Grayscaling and normalized"
[image15]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/newImages/01.jpg "Traffic Sign 1"
[image13]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/newImages/11.jpg "Traffic Sign 2"
[image16]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/newImages/14.jpg "Traffic Sign 3"
[image12]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/newImages/22.jpg "Traffic Sign 4"
[image14]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/newImages/25.jpg "Traffic Sign 5"
[image11]: /home/phuc/Study/Udacity/SeftDrivingCar/Term1/Project2/CarND-Traffic-Sign-Classifier-Project/newImages/40.jpg "Traffic Sign 6"

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

As a first step, I decided to convert the images to grayscale that shown in the classroom at the CNN lesson. Because I also think that the colors is not a factor in what classifies the sign structures    
Then, I normalized the gray image data using the suggested normalization method (pixel - 128)/ 128

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

It is well-known that depths of the layers are important in LeNet model.  
In order to keep the structure of the given LeNet example, I decide only to change the depths of layers by the multiply factor **dd**.   
I also found that the accuracy of the model was increased effectively by the factor **dd** (the depths of layers).  

I believe that some other optimizer or above parameters could increase the accuracy of my model. But it is not the key point of the LeNet method, so I want to close my optimization in here.



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I have run some times for the [Ipython](Traffic_Sign_Classifier.ipynb) notebook.   

Although the results of model were little changed, my final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.952
* test set accuracy of 0.933

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
