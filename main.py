import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from random import randint,random
df = pd.read_csv('titanic.csv')
#print(df[df['Fare'] == 0].head(20))
df.drop(['Cabin','Name','PassengerId','Ticket','Embarked'], axis = 1, inplace = True)
age_1 = df[df['Pclass'] == 1]['Age'].median()
age_2 = df[df['Pclass'] == 2]['Age'].median()
age_3 = df[df['Pclass'] == 3]['Age'].median()
def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age_1
        if row['Pclass'] == 2:
            return age_2
        return age_3
    return row['Age']

def zxc(a):
    if a == 'male':
        return 1
    return 0
def is_alone(row):
    if row['SibSp'] + row['Parch'] == 0:
        return 1
    return 0

df['Alone'] = df.apply(is_alone, axis = 1)
df['Age'] = df.apply(fill_age, axis = 1)
df['Sex'] = df['Sex'].apply(zxc)
"""print(df.info())
print(df['Fare'].agg(['min','max','mean']))
print(df.pivot_table(values = 'Age', columns = 'Alone', index = 'Survived', aggfunc = 'count'))"""
from time import time

def qwe(a, q):
    #s = time()
    x = df.drop('Survived', axis = 1)
    y = df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = a)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classifier = KNeighborsClassifier(n_neighbors = q)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    percent = accuracy_score(y_test, y_pred) * 100
    return percent, a, q
    #e = time()
    #print(e - s)    
    print(percent,a,q)

a1 = []
a2 = []
a3 = []
def max_1(a):
    z = max(a)
    return a.index(z)

def qqq(a,b,c):
    z = max_1(a)
    print(a[z],b[z],c[z])   
for i in range(100):
    a = qwe(randint(20,50)/100,list(range(1,6))[::2][randint(0,2)])
    a1.append(a[0])
    a2.append(a[1])
    a3.append(a[2])

qqq(a1,a2,a3)