import pandas as pd

def get_subset(df,start,end):
    assert (end - start) < 500, ValueError('subset should be shorter than 500')
    X = df['pixels'].iloc[start:end].str.split(expand=True).as_matrix()
    y = df['emotion'].iloc[start:end].values
    return X,y
