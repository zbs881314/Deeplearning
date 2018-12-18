from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.columns = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Class label']
#print(df.head())

X = df[['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']].values
#X=df.iloc[:,0:4]

y = pd.factorize(df['Class label'])[0]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

print('#Training data points:{}'.format(X_train.shape[0]))
print('#Testing data points:{}'.format(X_test.shape[0]))
print('Class label:{}(mapped from {})'.format(np.unique(y),np.unique(df['Class label'])))

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
#feature_columns = [tf.contrib.layers.real_valued_column(i) for i in df.columns[0:3]]
#print(feature_columns)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes = 3)
classifier.fit(x=X_train, y=y_train, steps=1000)
accuracy_score = classifier.evaluate(x=X_test,y=y_test)["accuracy"]
print('Accuracy:{0:f}'.format(accuracy_score))
