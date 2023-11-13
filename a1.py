import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pip._internal.utils.misc import tabulate
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import accuracy_score


penguins = pd.read_csv('penguins.csv')
print(penguins.head())

X = penguins.drop('species', axis=1)
Y = penguins['species']
X_dummies = pd.get_dummies(X)
print(X_dummies.head())

#df_penguins = pd.DataFrame(data=penguins)
#dummies = pd.get_dummies(df_penguins, columns="species")
X_train, X_test, Y_train, Y_test = train_test_split(X_dummies, Y, test_size=0.2, random_state=42)

model_penguins = DecisionTreeClassifier()
model_penguins.fit(X_train,Y_train)

Y_pred = model_penguins.predict(X_test)
print(Y_pred)
abalones = pd.read_csv('abalone.csv')
print(abalones.head())

A = abalones.drop('Type', axis=1)
B = abalones['Type']
A_dummies =pd.get_dummies(A)


#df_penguins = pd.DataFrame(data=penguins)
#dummies = pd.get_dummies(df_penguins, columns="species")
A_train, A_test, B_train, B_test = train_test_split(A_dummies, B, test_size=0.4, random_state=42)

model_abalones = DecisionTreeClassifier(max_depth=3)
model_abalones.fit(A_train,B_train)

B_pred = model_abalones.predict(A_test)

print(B_pred)

#penguins Base-DT
plt.figure("Desicion Tree: Penguins")
plt.tight_layout()
plot_tree(model_penguins, filled=True, fontsize=7)
plt.show()
cm = confusion_matrix(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average=None,zero_division=1)
recall = recall_score(Y_test, Y_pred, average=None,zero_division=1)
f1 = f1_score(Y_test, Y_pred, average=None,zero_division=1)
accuracy = accuracy_score(Y_test, Y_pred)
with open("penguin-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (BASE DT):\n")
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")


#abalone Base-DT
plt.figure("Desicion Tree: Abalone")
plt.tight_layout()
plot_tree(model_abalones, filled=True, fontsize=7)
plt.show()
cm = confusion_matrix(B_test, B_pred)
precision = precision_score(B_test, B_pred, average=None,zero_division=1)
recall = recall_score(B_test, B_pred, average=None,zero_division=1)
f1 = f1_score(B_test, B_pred, average=None,zero_division=1)
accuracy = accuracy_score(B_test, B_pred)
with open("abalone-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (BASE DT):\n")
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")


search_space = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5],
    'min_samples_split': [2, 5, 10]
}

#TOP DT abalone
top_dt_abalones = DecisionTreeClassifier()
grid_search_abalones = GridSearchCV(top_dt_abalones, search_space, cv=5)
grid_search_abalones.fit(A_train, B_train)
best_model = grid_search_abalones.best_estimator_
B_pred = best_model.predict(A_test)
top_dt_abalones = grid_search_abalones.best_estimator_
plt.figure("Top Decision Tree (Abalone)")
plt.tight_layout()
plot_tree(top_dt_abalones, filled=True, fontsize=7, max_depth=3)
plt.show()
cm = confusion_matrix(B_test, B_pred)
best_params = grid_search_abalones.best_params_
precision = precision_score(B_test, B_pred, average=None,zero_division=1)
recall = recall_score(B_test, B_pred, average=None,zero_division=1)
f1 = f1_score(B_test, B_pred, average=None,zero_division=1)
accuracy = accuracy_score(B_test, B_pred)
with open("abalone-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (TOP DT):\n" + str(best_params) + "\n")
    file.write("\n\n")
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")


#TOP DT Penguine
top_dt_penguins = DecisionTreeClassifier()
grid_search_penguins = GridSearchCV(top_dt_penguins, search_space, cv=5)
grid_search_penguins.fit(X_train, Y_train)
top_dt_penguins = grid_search_penguins.best_estimator_
Y_pred = top_dt_penguins.predict(X_test)
plt.figure("Top Decision Tree (Penguins)")
plt.tight_layout()
plot_tree(top_dt_penguins, filled=True, fontsize=7, max_depth=3)
plt.show()
cm = confusion_matrix(Y_test, Y_pred)
best_params = grid_search_penguins.best_params_
precision = precision_score(Y_test, Y_pred, average=None,zero_division=1)
recall = recall_score(Y_test, Y_pred, average=None,zero_division=1)
f1 = f1_score(Y_test, Y_pred, average=None,zero_division=1)
accuracy = accuracy_score(Y_test, Y_pred)
with open("penguin-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (TOP DT):\n" + str(best_params) + "\n")
    file.write("\n\n")
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")

#base MLP Penguine
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', random_state=50)
X_train, X_test, Y_train, Y_test = train_test_split(X_dummies, Y, test_size=0.2, random_state=50)
mlp.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'BASE MLP Accuracy (penguine): {accuracy}')
cm = confusion_matrix(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average=None,zero_division=1)
recall = recall_score(Y_test, Y_pred, average=None,zero_division=1)
f1 = f1_score(Y_test, Y_pred, average=None,zero_division=1)
accuracy = accuracy_score(Y_test, Y_pred)
with open("penguin-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (BASE MLP):\n")
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")

# Top_MLP Penguine
param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'solver': ['adam', 'sgd'],
    'max_iter': [600]
}
mlp = MLPClassifier(random_state=50)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy',error_score='raise')
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'TOP MLP Accuracy (penguin): {accuracy}')
cm = confusion_matrix(Y_test, Y_pred)
best_params = grid_search.best_params_
precision = precision_score(Y_test, Y_pred, average=None,zero_division=1)
recall = recall_score(Y_test, Y_pred, average=None,zero_division=1)
f1 = f1_score(Y_test, Y_pred, average=None,zero_division=1)
accuracy = accuracy_score(Y_test, Y_pred)
with open("penguin-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (TOP MLP):\n" + str(best_params) + "\n")
    file.write("\n\n")
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")
#base MLP abalone
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', random_state=50)
A_train, A_test, B_train, B_test = train_test_split(A_dummies, B, test_size=0.2, random_state=50)
mlp.fit(A_train, B_train)
B_pred = mlp.predict(A_test)
accuracy = accuracy_score(B_test, B_pred)
print(f'BASE MLP Accuracy (abalone): {accuracy}')
cm = confusion_matrix(B_test, B_pred)
precision = precision_score(B_test, B_pred, average=None,zero_division=1)
recall = recall_score(B_test, B_pred, average=None,zero_division=1)
f1 = f1_score(B_test, B_pred, average=None,zero_division=1)
accuracy = accuracy_score(B_test, B_pred)
with open("abalone-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (BASE MLP) :\n" )
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")
# Top_MLP abalone
param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'solver': ['adam', 'sgd'],
    'max_iter': [1200]
}
mlp = MLPClassifier(random_state=50)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy',error_score='raise')
grid_search.fit(A_train, B_train)
best_model = grid_search.best_estimator_
B_pred = best_model.predict(A_test)
accuracy = accuracy_score(B_test, B_pred)
best_params = grid_search.best_params_
print(f'TOP MLP Accuracy (abalone): {accuracy}')
cm = confusion_matrix(B_test, B_pred)
precision = precision_score(B_test, B_pred, average=None,zero_division=1)
recall = recall_score(B_test, B_pred, average=None,zero_division=1)
f1 = f1_score(B_test, B_pred, average=None,zero_division=1)
accuracy = accuracy_score(B_test, B_pred)
with open("abalone-performance.txt", "a") as file:
    file.write("\n\n")
    file.write("Confusion Matrix (TOP MLP):\n " + str(best_params) + "\n")
    file.write("\n\n")
    file.write("PRECISION: " + str(precision))
    file.write("\n\n")
    file.write("RECALL: " + str(recall))
    file.write("\n\n")
    file.write("F1: " + str(f1))
    file.write("\n\n")
    file.write("ACCURACY: " + str(accuracy))
    file.write("\n\n")
    np.savetxt(file, cm, fmt="%d", delimiter="\t")
    file.write("\n\n")
















