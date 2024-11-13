import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_PassengerId = test_df["PassengerId"]

train_df

train_df.columns

train_df.head(10)

train_df.tail(10)

train_df.describe()

train_df.info()

def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (12,4))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))

categorical_1 = ["Survived","Pclass","Sex","SibSp", "Parch","Embarked"]
for c in categorical_1:
    bar_plot(c)

categorical_2 = ["Name","Ticket","Cabin"]
for c in categorical_2:
    print("{} \n".format(train_df[c].value_counts()))

def plot_hist(var):
    plt.figure(figsize = (12,4))
    plt.hist(train_df[var], bins = 50)
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.title("{} Distribution with Hist".format(var))
    plt.show()

numeric = ["Fare", "Age","PassengerId"]
for n in numeric:
    plot_hist(n)

train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)

train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="SibSp", ascending = False)

train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)

def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers

train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)

train_df.columns[train_df.isnull().any()]

train_df.isnull().sum()

train_df[train_df["Embarked"].isnull()]

train_df.boxplot(column="Fare",by = "Embarked")
plt.show()

train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]

train_df[train_df["Age"].isnull()]

sns.heatmap(train_df[["Age","SibSp","Parch","Pclass"]].corr(), annot = True)
plt.show()

index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med

train_df[train_df["Age"].isnull()]

train_df['Title'] = train_df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
train_df['Is_Married'] = 0
train_df['Is_Married'].loc[train_df['Title'] == 'Mrs'] = 1

train_df.head(20)

fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
sns.barplot(x=train_df['Title'].value_counts().index, y=train_df['Title'].value_counts().values, ax=axs[0])

axs[0].tick_params(axis='x', labelsize=10)
axs[1].tick_params(axis='x', labelsize=15)

for i in range(2):    
    axs[i].tick_params(axis='y', labelsize=15)

axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)

train_df['Title'] = train_df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
train_df['Title'] = train_df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

sns.barplot(x=train_df['Title'].value_counts().index, y=train_df['Title'].value_counts().values, ax=axs[1])
axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)

plt.show()

train_df['Family_Size'] = train_df['SibSp'] + train_df['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
train_df['Family_Size_Grouped'] = train_df['Family_Size'].map(family_map)
train_df.head()

g = sns.catplot(x="Family_Size_Grouped", y="Survived", data=train_df, kind="bar")
g.set_ylabels("Survival")
plt.show()

sns.countplot(x = "Family_Size_Grouped", data = train_df)
plt.show()

g = sns.catplot(x = "Family_Size", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()

train_df['Ticket_Frequency'] = train_df.groupby('Ticket')['Ticket'].transform('count')

train_df.head()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

non_numeric_features = ['Embarked', 'Sex', 'Title', 'Family_Size_Grouped','Age', 'Fare']

label_encoder = LabelEncoder()

for column in non_numeric_features:
    train_df[column] = label_encoder.fit_transform(train_df[column])

cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Family_Size_Grouped']
one_hot_encoder = OneHotEncoder()
encoded_features = one_hot_encoder.fit_transform(train_df[cat_features]).toarray()

column_names = []
for i, column in enumerate(cat_features):
    unique_labels = train_df[column].unique()
    
    names = [f"{column}_{label}" for label in unique_labels]
    column_names.extend(names)

one_hot_encoded_df = pd.DataFrame(encoded_features, columns=column_names)

train_df = pd.concat([train_df, one_hot_encoded_df], axis=1)

train_df.head(20)

train_df.drop(labels = ["PassengerId", "Cabin", "Name", "Ticket"], axis = 1, inplace = True)

train_df.head()

train_df.columns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

test = train_df[train_df_len:]
test.drop("Survived", axis=1, inplace=True)

test

train = train_df[:train_df_len]
X_train = train.drop("Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test_df))

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier()
}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'{model_name}: Accuracy = {accuracy:.4f}')

logistic_regression = LogisticRegression()
logistic_regression = logistic_regression.fit(X_train, y_train)
print(accuracy_score(logistic_regression.predict(X_test),y_test))

test_survived = pd.Series(logistic_regression.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("submission.csv",header=True, index = False)

import nbformat

def extract_code_from_ipynb(ipynb_file, output_file):
    with open(ipynb_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)
        
    code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, code in enumerate(code_cells, 1):
            file.write(code)
            file.write('\n\n')

# Replace 'notebook.ipynb' and 'output.py' with your file names
extract_code_from_ipynb('titanic.ipynb', 'output.py')



