import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import filterwarnings
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

filterwarnings(action='ignore')

pd.set_option('display.max_columns', 15, 'display.width', 10000)
# LOADING DATASET
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# PRINTING DATASET
print("DATA:- \n", train_df.head(10))
print("\nSHAPE OF TRAIN_DATA:- \n", train_df.shape)
print("\nDESCRIPTION:- \n", train_df.describe())
total = train_df.isna().sum().sort_values(ascending=False)
percent_1 = round(train_df.isnull().sum() / train_df.isnull().count() * 100, 2)
miss = pd.concat([total, percent_1], axis=1, keys=['Total', 'Percentage'])
print("\nMISSING DATA:- \n", miss)
print("CORRELATION MATRIX:- ")
print(train_df.corr())


print("Survived:- ")
print(train_df['Survived'].value_counts())
print("Class vs Survived:- ")
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False))
print("Sex vs Survived:- ")
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print("SibSp vs Survived:- ")
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False))
print("Parch vs Survived:- ")
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False))
train_random_ages = np.random.randint(train_df["Age"].mean() - train_df["Age"].std(),
                                      train_df["Age"].mean() + train_df["Age"].std(),
                                      size=train_df["Age"].isnull().sum())
test_random_ages = np.random.randint(test_df["Age"].mean() - test_df["Age"].std(),
                                     test_df["Age"].mean() + test_df["Age"].std(),
                                     size=test_df["Age"].isnull().sum())
# Filling missing values
train_df["Age"][np.isnan(train_df["Age"])] = train_random_ages
test_df["Age"][np.isnan(test_df["Age"])] = test_random_ages
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)
train_df["Embarked"].fillna('S', inplace=True)
test_df["Embarked"].fillna('S', inplace=True)

a = train_df.loc[train_df['Sex'] == 'male']
b = train_df.loc[train_df['Sex'] == 'female']
c = a.loc[train_df['Survived'] == 1, 'Age']
d = b.loc[train_df['Survived'] == 1, 'Age']
e = a.loc[train_df['Survived'] == 0, 'Age']
f = b.loc[train_df['Survived'] == 0, 'Age']
kwargs = dict(alpha=0.5, bins=30)
kwargs1 = dict(alpha=0.5, bins=40)

with PdfPages(r'C:\Users\KIIT\Desktop\titanic prediction.pdf') as export_pdf:
    sns.heatmap(train_df.corr(), annot=True)
    export_pdf.savefig()
    plt.close()


    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.15, 0.4, 0.6])
    ax.set_title("MEN")
    ax.set_xlabel("Age")
    ax.hist(c, **kwargs, color='g', label='Survived')
    ax.hist(e, **kwargs, color='b', label='Died')
    ax.legend()
    bx = fig.add_axes([0.55, 0.15, 0.4, 0.6])
    bx.set_title("WOMEN")
    bx.set_xlabel("Age")
    bx.hist(d, **kwargs1, color='g', label='Survived')
    bx.hist(f, **kwargs1, color='b', label='Died')
    bx.legend()
    export_pdf.savefig()
    plt.close()

    plt.scatter(train_df['Fare'], train_df['Pclass'], color='purple', label='Passenger Paid')
    plt.ylabel('Pclass')
    plt.xlabel('Price / Fare')
    plt.title('Price Of Each Class')
    plt.legend()
    export_pdf.savefig()
    plt.close()

    plt.scatter(train_df['Age'], train_df['Survived'], color='g', marker='*')
    plt.ylabel('Survived')
    plt.xlabel('Age')
    plt.title('SURVIVAL')
    plt.legend()
    export_pdf.savefig()
    plt.close()
    train_df.hist()
    export_pdf.savefig()
    plt.close()

    train_df.plot(kind='density', sharex=False, subplots=True, layout=(4, 2))
    export_pdf.savefig()
    plt.close()

    train_df.plot(kind='box', subplots=True, sharex=True, layout=(2, 4))
    export_pdf.savefig()
    plt.close()

    from pandas.plotting import scatter_matrix
    scatter_matrix(train_df)
    export_pdf.savefig()
    plt.close()

    sns.barplot(x='Pclass', y='Survived', data=train_df)
    export_pdf.savefig()
    plt.close()

    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()
    export_pdf.savefig()
    plt.close()

    sns.countplot(train_df['Survived'], label="Count")
    export_pdf.savefig()
    plt.close()

    plt.subplot(231)
    plt.title("Sex")
    sns.countplot(train_df['Sex'],hue=train_df['Survived'])
    plt.subplot(232)
    plt.title("Pclass")
    sns.countplot(train_df['Pclass'],hue=train_df['Survived'])
    plt.subplot(233)
    plt.title("SibSp")
    sns.countplot(train_df['SibSp'],hue=train_df['Survived'])
    plt.subplot(234)
    plt.title("Parch")
    sns.countplot(train_df['Parch'],hue=train_df['Survived'])
    plt.subplot(235)
    plt.title("Embarked")
    sns.countplot(train_df['Embarked'],hue=train_df['Survived'])
    export_pdf.savefig()
    plt.close()

data = [train_df, test_df]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
    dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
print("THE TEST DATA WHICH NEEDS TO BE ENCODED:- ")
print(test_df.head(5))
print("VALUES BEFORE ENCODING:- ")
print("SEX:- ", test_df['Sex'].unique())
print("EMBARKED:- ", test_df['Embarked'].unique())

# Encoding categorical data values (Transforming object data types to integers)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

# Encode sex column
test_df.iloc[:, 2] = labelencoder.fit_transform(test_df.iloc[:, 2].values)
train_df.iloc[:, 2] = labelencoder.fit_transform(train_df.iloc[:, 2].values)

# Encode embarked
test_df.iloc[:, 7] = labelencoder.fit_transform(test_df.iloc[:, 7].values)
train_df.iloc[:, 7] = labelencoder.fit_transform(train_df.iloc[:, 7].values)
# Print the NEW unique values in the columns
print("VALUES AFTER ENCODING:- ")
print(test_df['Sex'].unique())
print(test_df['Embarked'].unique())
print("THE TEST DATA LOOKS LIKE:- ")
print(test_df.head(10))

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop('PassengerId', axis=1).copy()

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred1 = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


nb = GaussianNB()
nb.fit(X_train, Y_train)
y_pred2 = nb.predict(X_test)
acc_nb = round(nb.score(X_train, Y_train) * 100, 2)

svm = SVC()
svm.fit(X_train, Y_train)
y_pred3 = svm.predict(X_test)
acc_svm = round(svm.score(X_train, Y_train) * 100, 2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred4 = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

dcs = DecisionTreeClassifier(criterion='entropy', random_state=7)
dcs.fit(X_train, Y_train)
y_pred5 = dcs.predict(X_test)
acc_dcs = round(dcs.score(X_train, Y_train) * 100, 2)

res = pd.DataFrame({'MODEL': ['Logistic Regression', 'SVM', 'KNN', 'Naive-Bayes', 'Decision Tree'],
                    'TRAINING SCORES': [acc_log, acc_svm, acc_knn, acc_nb, acc_dcs]})
print(res)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score

prediction1 = cross_val_predict(logreg, X_train, Y_train, cv=3)
prediction2 = cross_val_predict(svm, X_train, Y_train, cv=3)
prediction3 = cross_val_predict(nb, X_train, Y_train, cv=3)
prediction4 = cross_val_predict(knn, X_train, Y_train, cv=3)
prediction5 = cross_val_predict(dcs, X_train, Y_train, cv=3)
score1 = round(cross_val_score(logreg, X_train, Y_train, cv=3).mean() * 100, 2)
score2 = round(cross_val_score(svm, X_train, Y_train, cv=3).mean() * 100, 2)
score3 = round(cross_val_score(nb, X_train, Y_train, cv=3).mean() * 100, 2)
score4 = round(cross_val_score(knn, X_train, Y_train, cv=3).mean() * 100, 2)
score5 = round(cross_val_score(dcs, X_train, Y_train, cv=3).mean() * 100, 2)
res1 = pd.DataFrame({'MODEL': ['Logistic Regression', 'SVM', 'KNN', 'Naive-Bayes', 'Decision Tree'],
                     'VALIDATION SCORES': [score1, score2, score3, score4, score5]})
print(res1)
print("CONFUSION MATRIX OF LOGISTIC REGRESSION:- ")
print(confusion_matrix(Y_train, prediction1))
print("CONFUSION MATRIX OF KNN:- ")
print(confusion_matrix(Y_train, prediction4))
print("CONFUSION MATRIX OF SVM:- ")
print(confusion_matrix(Y_train, prediction2))
print("CONFUSION MATRIX OF NAIVE_BAYES:- ")
print(confusion_matrix(Y_train, prediction3))
print("CONFUSION MATRIX OF DECISION TREE:- ")
print(confusion_matrix(Y_train, prediction5))
print("CLASSIFICATION REPORT OF LOGISTIC REGRESSION:- ")
print(classification_report(Y_train, prediction1))
print("CLASSIFICATION REPORT OF SVM:- ")
print(classification_report(Y_train, prediction2))
print("CLASSIFICATION REPORT OF NAIVE_BAYES:- ")
print(classification_report(Y_train, prediction3))
print("CLASSIFICATION REPORT OF KNN:- ")
print(classification_report(Y_train, prediction4))
print("CLASSIFICATION REPORT OF DECISION TREE:- ")
print(classification_report(Y_train, prediction5))

submit = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                       'Survived': y_pred5})

submit.to_csv('Survival.csv', index=False)
print("EXPORTED")
df = pd.read_csv('Survival.csv')
print("Predicted Survival:- \n 1 for survived \n 0 for not survived:- ")
print(df['Survived'].value_counts())

print("THE BEST TRAINING MODEL IS THE DECISION TREE. LET'S NOW PREDICT WHETHER YOU WILL SURVIVE OR NOT:- ")
pc = int(input("ENTER THE PASSENGER CLASS(1 or 2 or 3):- "))
s = int(input("ENTER YOUR GENDER(1 for male and 0 for female):- "))
ag = int(input("ENTER YOUR AGE:- "))
ss = int(input("ENTER NO. OF SIBLINGS OR SPOUSES:- "))
par = int(input("ENTER NO. OF PARENTS OR CHILDREN:- "))
f = int(input("ENTER FARE AMOUNT(0-5 low to high):- "))
e = int(input("ENTER EMBARKED(1,2 or 0):- "))
my_survival = [[pc, s, ag, ss, par, f, e]]
# Print Prediction of Random Forest Classifier model
pred = dcs.predict(my_survival)
print("SURVIVED:- 1 \t DIED:- 0 \nPREDICTION AS PER YOUR INPUT:- {}".format(pred))

if pred == 0:
    print("Oh no! You didn't make it")
else:
    print('Nice! You survived')


