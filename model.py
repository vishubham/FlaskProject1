import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pickle

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace(to_replace= [0, 1, 2], value = ['setosa', 'versicolor', 'virginica'])
print('df.head():\n', df.head())

# Select dependant and independent variables
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
print('X.head():\n', X.head())
print('y.head():\n', y.head())

# Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# Scale the features
stdscaler = StandardScaler()
X_train = stdscaler.fit_transform(X_train)
X_test = stdscaler.transform(X_test)

# Model instantiation
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make a pickle file for the trained model
pickle.dump(classifier, open('rfcmodel.pkl', 'wb'))
