# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./label_counts.jpg "Visualization"
[image2]: ./web_images/12.jpg "Traffic Sign 1"
[image3]: ./web_images/13.jpg "Traffic Sign 2"
[image4]: ./web_images/26.jpg "Traffic Sign 3"
[image5]: ./web_images/31.jpg "Traffic Sign 4"
[image6]: ./web_images/39.jpg "Traffic Sign 5"



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1.  Preprocess the image data.

- shuffle training data
- grayscaling
- normalization


#### 2. Model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray image   		|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 |
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 		|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64 |
| Max pooling	| 2x2 stride,  outputs 8x8x64 |
| Convolution 3x3 | 1x1 stride, same padding, outputs 8x8x256  |
|   Max pooling   | 2x2 stride,  outputs 4x4x64 |
| Flatten | outputs 1024 |
| Dropout | p = 0.5 |
| Fully connected | outputs 256, apply ReLU |
| Dropout | p=0.4 |
| Fully connected | outputs 43 |



#### 3. Describe how I trained my model. 

To train the model, I used an Adam optimizer with initial learning rate  3e-4 and decay rate 0.9 every 1000 step.

The bath size is 32. The maximum number of epochs is 30.

#### 4. The approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 93.4%
* test set accuracy of 92.4%

An iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  I choose the architecture like Lenet first.  

* What were some problems with the initial architecture?

  Overfitting is obvious. It works well on train set but perform poorly on validation set.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  I add two dropout layers to reduce overfitting.

* Which parameters were tuned? How were they adjusted and why?

  The number of neuron of each hidden layer is increased because the final class number 43 is lager than 10, so a larger network is required.  Learning rate and dropout rate are also tuned to get solution faster.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  Dropout is a very important design choice, it significantly reduce the overfitting and allow me getting a 90% accuracy on validation just in several epochs.  


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6]

The third image might be difficult to classify because it is similar to General caution sign in gray scale.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road | Priority road |
| Yield   | Yield 					|
| Traffic signals	| Traffic signals	|
| Wild animals crossing | Wild animals crossing |
| Keep left	| Keep left    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.4%.

For the third image, the model is relatively sure that this is a Traffic signals sign (probability of 0.768), and the image does contain a Traffic signals sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.768        | Traffic signals |
| 0.232    | General caution |
| 0.000 (8e-11)	| Pedestrians	|
| 0.000(5e-13)	| Right-of-way at the next intersection |
| 0.000(2e-15)	| Road narrows on the right |

For other 4 images, the model is relatively sure each prediction with a probability greater than 99.9%