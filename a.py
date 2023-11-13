import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree  
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


data_penguins = pd.read_csv('penguins.csv')
data_abalone = pd.read_csv('abalone.csv')

penguins = pd.DataFrame(data_penguins)
abalone = pd.DataFrame(data_abalone)

penguins_encoded = pd.get_dummies(penguins, columns=['island'])

sex_mapping = {'MALE': 0, 'FEMALE': 1}
penguins_encoded['sex'] = penguins_encoded['sex'].map(sex_mapping)

class_counts = penguins_encoded['species'].value_counts()
class_counts.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Penguin Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')
plt.savefig('penguin-classes.jpg')
plt.show()

X = penguins_encoded.drop('species', axis=1)
Y = penguins_encoded['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class_counts.plot(kind='bar', color=['blue', 'green', 'pink'])
plt.title('Abalone Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.savefig('abalone-classes.jpg')
plt.show()

A = abalone.drop('Type', axis=1)
B = abalone['Type']
A_dummies =pd.get_dummies(A)
B_dummies = pd.get_dummies(B)


#df_penguins = pd.DataFrame(data=penguins)
#dummies = pd.get_dummies(df_penguins, columns="species")
A_train, A_test, B_train, B_test = train_test_split(A_dummies, B_dummies, test_size=0.2, random_state=42)

model_abalones = DecisionTreeClassifier(max_depth=3)
model_abalones.fit(A_train,B_train)

B_pred = model_abalones.predict(A_test)


# Base Decision Tree model
model_abalone = DecisionTreeClassifier(max_depth=3)
model_penguins = DecisionTreeClassifier()

# Fit the model to the training data
'''model_abalone.fit(x_train, y_train)
y_pred = model_abalone.predict(x_test)
'''
model_penguins.fit(X_train, Y_train)
Y_pred = model_penguins.predict(X_test)


# Visualize the ABALONE tree 
plt.figure("Base Desicion Tree: Abalone")
plt.tight_layout()
plot_tree(model_abalones, filled=True, fontsize=7)
plt.show()

# Visualize the PENGUIN tree 
plt.figure("Base Decision Tree (Penguin)")
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
top_dt_abalones = DecisionTreeClassifier()

top_dt_penguins = DecisionTreeClassifier()

# grid search to find the best hyperparameters
grid_search_abalones = GridSearchCV(top_dt_abalones, search_space, cv=5)
grid_search_abalones.fit(A_train, B_train)

# best model from grid search
top_dt_abalones = grid_search_abalones.best_estimator_

plt.figure("Top Decision Tree (Abalone)")
plt.tight_layout()
plot_tree(top_dt_abalones, filled=True, fontsize=7, max_depth=3)
plt.show()

# grid search to find the best hyperparameters
grid_search_penguins = GridSearchCV(top_dt_penguins, search_space, cv=5)
grid_search_penguins.fit(A_train, B_train)

# best model from grid search
top_dt_penguins = grid_search_penguins.best_estimator_

plt.figure("Top Decision Tree (Penguins)")
plt.tight_layout()
plot_tree(top_dt_penguins, filled=True, fontsize=7, max_depth=3)
plt.show()


'''#evaluate the model
# f1
labels_order=[0,1,2,3]
f1 = f1_score(y_true = y_test, y_pred=y_pred, labels=labels_order, average="weighted", zero_division=1)
print(f"f1: {f1}")

#accuracy
acc= accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy: {acc}")'''
