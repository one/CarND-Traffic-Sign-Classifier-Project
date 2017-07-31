# **Traffic Sign Recognition with artificial neural networks** 

### Writeup by Hannes Bergler

---

The goals / steps of this project were the following:
* Load the dataset (see below for links to the project dataset)
* Explore, summarize and visualize the dataset
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/training_data_histogram.png "Visualization"
[image2]: ./images/downloaded_32x32/img0.png "downloaded image 1"
[image3]: ./images/downloaded_32x32/img1.png "downloaded image 2"
[image4]: ./images/downloaded_32x32/img2.png "downloaded image 3"
[image5]: ./images/downloaded_32x32/img3.png "downloaded image 4"
[image6]: ./images/downloaded_32x32/img4.png "downloaded image 5"
[image7]: ./images/downloaded_32x32/img5.png "downloaded image 6"
[image8]: ./images/downloaded_32x32/img6.png "downloaded image 7"
[image9]: ./images/downloaded_32x32/img7.png "downloaded image 8"
[image10]: ./images/downloaded_32x32/img8.png "downloaded image 9"
[image11]: ./images/downloaded_32x32/img9.png "downloaded image 10"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### I. Submission Files

The submission includes a writeup, which you're reading right now!

And here is a link to my [project code](https://github.com/one/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### II. Dataset Summary & Exploration

#### 1. Basic summary of the dataset.

Summary statistics of the traffic signs dataset:

* The size of training set is *34799*
* The size of the validation set is *4410*
* The size of test set is *12630*
* The shape of a traffic sign image is *(32, 32, 3)*
* The number of unique classes/labels in the dataset is *43*

#### 2. Exploratory visualization of the dataset.

Here is a bar chart showing how the traffic sign classes are distributed in the training dataset. You can see that some classes (e.g. 1 and 2) are much more common than others (e.g. 0 and 19).

![histogram][image1]

For visualizing the dataset, I also printed out one image of each traffic sign class in the [jupyter notebook](https://github.com/one/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html).

### III. Design and Test a Model Architecture

#### 1. Preprocessing the image data.

For preprocessing the data I used the following steps:
- shuffle the training data, to get a random order of the images
- normalization of the image data: [0 .. 255] --> [0.1 .. 0.9]

I did NOT convert the images to grayscale to not lose the color information. I found that with my setup the prediction accuracy of the validation dataset dropped by 0.01, when converting the images to grayscale.


#### 2. Final model architecture.

I used the LeNet architecture as a starting point, which works very well on 32x32 images. To adapt it to the colored images and the larger number of output classes, I doubled the size of every network layer.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 2x2 stride, VALID padding, outputs 28x28x12 	|
| RELU					|						simple activation function						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 2x2 stride, VALID padding, outputs 10x10x32 	|
| RELU					|						simple activation function						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten  |    change output shape from 5x5x32 to 800 |
| Fully connected with dropout		|   output: 240									|
| RELU					|						simple activation function						|
| Fully connected	with dropout	|   output: 168									|
| RELU					|						simple activation function						|
| Output layer |   output: 43   |

#### 3. Training the model.

For training the model, I used the following parameters:

- optimizer: AdamOptimizer (tensorflow)
- batch size: 128
- number of epochs: 20
- learning rate: 0.0005
- dropout rate: 0.5


#### 4. The approach for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* validation set accuracy of 0.960
* test set accuracy of 0.944

I started off with the standard LeNet architecture from class, because this architecture is able to classify 32p by 32p images quite good by default (as discussed in class).
With standard LeNet, I reached a validation set accuracy of about 0.89.

To improve the accuracy, I added dropout to the fully connected layers of the network. I found a dropout rate of 0.5 to be the optimum for this architecture.
I also doubled the size of each layer in the network to match the fact that the number of output classes in the German traffic sign dataset (n_classes = 43) is much higher than in the MNIST dataset (n_classes = 10). And also to match the fact that there is more information in the colored traffic sign images than in the grayscale images of the MNIST dataset.

The validation set accuracy of 0.96 - which is 0.03 points higher than the minimum expectation - shows that the model works well.

### IV. Test a Model on New Images

#### 1. Additional German traffic signs found on the web.

Here are not five but ten German traffic signs that I found on the web:

![traffic sign image][image2] ![traffic sign image][image3] ![traffic sign image][image4] ![traffic sign image][image5] ![traffic sign image][image6]
![traffic sign image][image7] ![traffic sign image][image8] ![traffic sign image][image9] ![traffic sign image][image10] ![traffic sign image][image11]

The first image might be difficult to classify because is shows the traffic sign a litte bit from the side, so the image is a bit distorted. The other nine images should be easyer to classify.

#### 2. Predictions on these new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| (1) Speed limit (30km/h)	| (1) Speed limit (30km/h)	| 
| (1) Speed limit (30km/h)	| (1) Speed limit (30km/h)	| 
| (1) Speed limit (30km/h)	| (1) Speed limit (30km/h)	| 
| (2) Speed limit (50km/h)	| (2) Speed limit (50km/h)	|
| (11)	Right-of-way at the next intersection	| (11) Right-of-way at the next intersection	|
| (12)	Priority road		| (12) Priority road		|
| (40)	Roundabout mandatory| (40) Roundabout mandatory	|
| (9) No passing			| (9) No passing   			|
| (9) No passing			| (9) No passing   			|
| (2) Speed limit (50km/h)	| (2) Speed limit (50km/h)	|

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.
TODO

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
