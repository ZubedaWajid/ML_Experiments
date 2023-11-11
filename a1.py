import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree  
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


penguins = pd.read_csv('penguins.csv')
abalone = pd.read_csv('abalone.csv')


penguins_encoded = pd.get_dummies(penguins, columns= ['sex'])
'''
sex_mapping = {'MALE': 0, 'FEMALE': 1}
penguins['sex'] = penguins['sex'].map(sex_mapping)'''

island_mapping = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
penguins['island'] = penguins['island'].map(island_mapping)

class_counts = penguins['species'].value_counts()
class_counts.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Penguin Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.savefig('penguin-classes.jpg')
plt.show()

X = penguins.drop('species', axis=1)
Y = penguins['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class_counts = abalone['Type'].value_counts()
class_counts.plot(kind='bar', color=['blue', 'green', 'pink'])
plt.title('Abalone Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.savefig('abalone-classes.jpg')
plt.show()


x = abalone.drop('Type', axis=1)  
y = abalone['Type']  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)





# Base Decision Tree model
model_abalone = DecisionTreeClassifier(max_depth=3)
model_penguins = DecisionTreeClassifier()

# Fit the model to the training data
model_abalone.fit(x_train, y_train)
y_pred = model_abalone.predict(x_test)

model_penguins.fit(X_train, Y_train)
Y_pred = model_penguins.predict(X_test)



#evaluate the model
# f1
labels_order=[0,1,2,3]
f1 = f1_score(y_true = y_test, y_pred=y_pred, labels=labels_order, average="weighted", zero_division=1)
print(f"f1: {f1}")

#accuracy
acc= accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy: {acc}")

# Visualize the ABALONE tree 
plt.figure("Desicion Tree: Abalone")
plt.tight_layout()
plot_tree(model_abalone, filled=True, fontsize=7)
plt.show()

# Visualize the PENGUIN tree 
plt.figure("Decision Tree (Penguin)")
plt.tight_layout()
plot_tree(model_penguins, filled=True, fontsize=7)
plt.show()

# hyperparameters to search
search_space = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5],  
    'min_samples_split': [2, 5, 10] 
}

#Decision Tree classifier
top_dt = DecisionTreeClassifier()

# grid search to find the best hyperparameters
grid_search = GridSearchCV(top_dt, search_space, cv=5)
grid_search.fit(x_train, y_train)

# best model from grid search
top_dt = grid_search.best_estimator_

plt.figure("Top Decision Tree (Abalone)")
plt.tight_layout()
plot_tree(top_dt, filled=True, fontsize=7, max_depth=3)
plt.show()