import pandas as pd

def get_subset(df,start=None,end=None,indices=None):  # no more use!
	# print(start is not None,end is not None,indices is not None)
	assert (start is None or end is None) and indices is not None, ValueError('can only pass indices alone')
	if start is not None and end is not None:
		assert (end - start) < 500, ValueError('subset should be shorter than 500')

	if indices is None:
		X = df['pixels'].iloc[start:end].str.split(expand=True).as_matrix()
		y = df['emotion'].iloc[start:end].values
		return X,y
	elif indices is not None:
		X = df['pixels'].iloc[indices].str.split(expand=True).as_matrix()
		y = df['emotion'].iloc[indices].values
		return X,y
	else:
		return None

def get_emotion_data():
    X_train = pd.read_csv('data/X_train.csv',header=None,dtype=np.float64).as_matrix()
    y_train = pd.read_csv('data/y_train.csv',header=None,dtype=np.float64).as_matrix()
    X_val = pd.read_csv('data/X_val.csv',header=None,dtype=np.float64).as_matrix()
    y_val = pd.read_csv('data/y_val.csv',header=None,dtype=np.float64).as_matrix()
    X_test = pd.read_csv('data/X_test.csv',header=None,dtype=np.float64).as_matrix()
    y_test = pd.read_csv('data/y_test.csv',header=None,dtype=np.float64).as_matrix()
    
    mean_image = np.mean(X_train,axis=0)
    plt.imshow(mean_image.reshape(48,48))
    
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    return X_train, y_train, X_val, y_val, X_test, y_test
