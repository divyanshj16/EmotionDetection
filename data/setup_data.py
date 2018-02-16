import pandas as  pd

print('Extracting data...\n')

df = pd.read_csv('train.csv')
df_train = df[df['Usage'] == 'Training']
df_val = df[df['Usage'] == 'PublicTest']
df_test = df[df['Usage'] == 'PrivateTest']

print('writing train data to file...(X_train.csv/y_train.csv)')
with open('X_train.csv','w') as file:
    for e in df_train['pixels']:
        print(','.join(e.split()),file=file)
with open('y_train.csv','w') as file:
    for e in df_train['emotion']:
        print(e,file=file)

print('writing validation data to file...(X_val.csv/y_val.csv)')
with open('X_val.csv','w') as file:
    for e in df_val['pixels']:
        print(','.join(e.split()),file=file)
with open('y_val.csv','w') as file:
    for e in df_val['emotion']:
        print(e,file=file)

print('writing test data to file...(X_test.csv/y_test.csv)')
with open('X_test.csv','w') as file:
    for e in df_test['pixels']:
        print(','.join(e.split()),file=file)
with open('y_test.csv','w') as file:
    for e in df_test['emotion']:
        print(e,file=file)

print('\n\ndone!')