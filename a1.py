import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

penguins = pd.read_csv('penguins.csv')
abalone = pd.read_csv('abalone.csv')


penguins = pd.get_dummies(penguins, columns=['island'], prefix='island')

sex_mapping = {'Male': 0, 'Female': 1}
penguins['sex'] = penguins['sex'].map(sex_mapping)

class_counts = penguins['species'].value_counts()
class_counts.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Penguin Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.savefig('penguin-classes.jpg')
plt.show()

class_counts = abalone['Type'].value_counts()
class_counts.plot(kind='bar', color=['blue', 'green', 'pink'])
plt.title('Abalone Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.savefig('abalone-classes.jpg')
plt.show()


X = abalone.drop('Type', axis=1)  
y = abalone['Type']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
