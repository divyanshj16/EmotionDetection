import pandas as pd

def get_subset(df,start=None,end=None,indices=None):
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
