import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree  
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


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


x = abalone.drop('Type', axis=1)  #axis=1 because we want to drop the column 
y = abalone['Type']  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)





# Create the Base Decision Tree model
model = DecisionTreeClassifier(max_depth=3)

# Fit the model to the training data
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#evaluate the model
# f1
labels_order=[0,1,2,3]
f1 = f1_score(y_true = y_test, y_pred=y_pred, labels=labels_order, average="weighted")
print(f"f1: {f1}")

#accuracy
acc= accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy: {acc}")

# Visualize the tree
plt.figure("Desicion Tree")
plt.tight_layout()
plot_tree(model, filled=True, fontsize=7)
plt.show()
