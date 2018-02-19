# Download data from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
# and put it into data/ directory
tar -xf fer2013.tar
mv fer2013/fer2013.csv ./train.csv
rm fer2013.tar
rm -rf fer2013
