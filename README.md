# Facial Expression Classification 

Emotion/Expression detection is fairly easy task for humans. Here I will try to predict different expressions using machine learning models.

The dataset is from a [kaggle competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) which was hosted in 2013. 

#### Data Description

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column. There is a 'Usage' which contains either 'Training' or 'Testing' labels to divide the dataset into train and test sets.

The training set consists of 28,709 examples. 

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of their ongoing research project(2013).