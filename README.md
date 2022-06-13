# Age_Gender_Prediction
This is a face recognition/classification project that uses Convolutional Neural Networks (CNN) to predict the user's age, gender, and ethnicity.
Try it yourself on : https://who-r-u.herokuapp.com/

The user is required to upload a clear picture of only their face and the model predicts the user's age, gender, and ethnicity

The model is a Convolutional Neural Network and the dataset used to train the model is the UTKFace dataset (https://www.kaggle.com/datasets/jangedoo/utkface-new/code) which consists of over 20,000 face images. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc.

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. This is the most widely used network in computer vision.

dataProcessing.py
1. The first part of this file cleans and preprocesses the data. Age labels are divided into 5 different categories (0-20, 21-40, 41-60, 61-80, 80+) and then are one-hot encoded. The pixels of the data are converted into a 48x48 numpy array grayscale image and is normalized.
2. The second part splits the data into train, test, and validation datasets (60-20-20 split).
3. The third part builds the CNN for gender classification. This model consists of 3 convolutional layers of sizes 64, 128, 256 respectively, and 2 fully connected layers of sizes 128 and 64 respectively. There is a max pooling layer of pool size (2,2) after each convolutional layer. The kernel size of the convolutional layers are (3,3) with same padding and stride length 1. The loss function used for this model is the bianry cross-entropy function because the output is binary (0 for male and 1 for female). The activation function used for every layer except the last is relu. Sigmoid activation is used for the last layer to ensure that there are no dead neurons. The optimizer used is adam.
4. For the age and ethnicity model, all the parameters are the same except that the output layer uses softmax activation function. Also the cost function used is mean squared error because calculating age is a regression task.

Overall, the accuracies obtained are:
Age - training = 87%, testing = 79%
Gender - training = 97%, testing = 91%


