import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #accuracy measure
from sklearn.metrics import recall_score, auc, precision_recall_curve, precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import svm #support vector Machine
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.linear_model import LogisticRegression #logistic regression


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('creditcard.csv')

def geometric_mean(precision, recall):
    return (precision * recall) ** 0.5

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

print('Distribution of the Classes in the dataset')
print(df['Class'].value_counts())
print(df['Class'].value_counts()/len(df))

print("---------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------RANDOM_FOREST----------------------------------------------------
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
prediction1=model.predict(X_test)

print("Random Forest AC score: ", round(metrics.accuracy_score(prediction1, y_test), 4))

f1_rf = f1_score(y_test, prediction1, average='weighted')  # 'weighted' berie do úvahy váhy tried pri výpočte
print("Random Forest F1 score:", round(f1_rf, 4))

# Výpočet recall metriky
recall_rf = recall_score(y_test, prediction1, average='weighted')  # 'weighted' berie do úvahy váhy tried pri výpočte
print("Random Forest Recall score:", round(recall_rf, 4))

precision_rf = precision_score(y_test, prediction1)
print("Random Forest Precision score:", round(precision_rf, 4))

# Výpočet geometrického priemeru
geo_mean_rf = geometric_mean(precision_rf, recall_rf)

print("Random Forest Geometric Mean score:", round(geo_mean_rf, 4))

# Výpočet AUC-PR
y_scores_rf = model.predict_proba(X_test)  # Pravdepodobnosti pre jednotlivé triedy
precision, recall, _ = precision_recall_curve(y_test, y_scores_rf[:, 1])
auc_pr_rf = auc(recall, precision)
print("Random Forest AUC-PR score:", round(auc_pr_rf, 4))


print("---------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------LINEAR_SVM-------------------------------------------------------
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1, probability=True)
model.fit(X_train,y_train)
prediction2=model.predict(X_test)

print("Linear SVM AC score: ",round(metrics.accuracy_score(prediction2, y_test), 4))

f1_svm = f1_score(y_test, prediction2, average='weighted')
print("Linear SVM F1 score:", round(f1_svm, 4))

recall_svm = recall_score(y_test, prediction2, average='weighted')
print("Linear SVM Recall score:", round(recall_svm, 4))

precision_svm = precision_score(y_test, prediction2)
print("Linear SVM Precision score:", round(precision_svm, 4))

geo_mean_svm = geometric_mean(precision_svm, recall_svm)

print("Linear SVM Geometric Mean score:", round(geo_mean_svm, 4))

y_scores_svm = model.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_scores_svm[:, 1])
auc_pr_svm = auc(recall, precision)
print("Linear SVM AUC-PR score:", round(auc_pr_svm, 4))


print("---------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------NON_LINEAR_SVM---------------------------------------------------
model=svm.SVC(kernel='rbf',C=1,gamma=0.1,probability=True)
model.fit(X_train,y_train)
prediction3=model.predict(X_test)

print('Non-linear SVM AC score: ',round(metrics.accuracy_score(prediction3, y_test), 4))

f1_nsvm = f1_score(y_test, prediction3, average='weighted')
print("Non-linear SVM F1 score:", round(f1_nsvm, 4))

recall_nsvm = recall_score(y_test, prediction3, average='weighted')
print("Non-linear SVM Recall score:", round(recall_nsvm, 4))

precision_nsvm = precision_score(y_test, prediction3)
print("Non-linear SVM Precision score:", round(precision_nsvm, 4))

geo_mean_nsvm = geometric_mean(precision_nsvm, recall_nsvm)

print("Non-linear SVM Geometric Mean score:", round(geo_mean_nsvm, 4))

y_scores_nsvm = model.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_scores_nsvm[:, 1])
auc_pr_nsvm = auc(recall, precision)
print("Non-linear SVM AUC-PR score:", round(auc_pr_nsvm, 4))


print("---------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------DECISION_TREE----------------------------------------------------
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction4=model.predict(X_test)

print("Decision Tree AC score: ",round(metrics.accuracy_score(prediction4, y_test), 4))

f1_dt = f1_score(y_test, prediction4, average='weighted')
print("Decision Tree F1 score:", round(f1_dt, 4))

recall_dt = recall_score(y_test, prediction4, average='weighted')
print("Decision Tree Recall score:", round(recall_dt, 4))

precision_dt = precision_score(y_test, prediction4)
print("Decision Tree Precision score:", round(precision_dt, 4))

geo_mean_dt = geometric_mean(precision_dt, recall_dt)

print("Decision Tree Geometric Mean score:", round(geo_mean_dt, 4))

y_scores_dt = model.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_scores_dt[:, 1])
auc_pr_dt = auc(recall, precision)
print("Decision Tree AUC-PR score:", round(auc_pr_dt, 4))


print("---------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------GAUSSIAN_NAIVE_BAYES---------------------------------------------
model=GaussianNB()
model.fit(X_train,y_train)
prediction5=model.predict(X_test)

print("Gaussian Naive Bayes AC score: ",round(metrics.accuracy_score(prediction5, y_test), 4))

f1_gnb = f1_score(y_test, prediction5, average='weighted')
print("Gaussian Naive Bayes F1 score:", round(f1_gnb, 4))

recall_gnb = recall_score(y_test, prediction5, average='weighted')
print("Gaussian Naive Bayes Recall score:", round(recall_gnb, 4))

precision_gnb = precision_score(y_test, prediction5)
print("Gaussian Naive Bayes Precision score:", round(precision_gnb, 4))

geo_mean_gnb = geometric_mean(precision_gnb, recall_gnb)

print("Gaussian Naive Bayes Geometric Mean score:", round(geo_mean_gnb, 4))

y_scores_gnb = model.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_scores_gnb[:, 1])
auc_pr_gnb = auc(recall, precision)
print("Gaussian Naive Bayes AUC-PR score:", round(auc_pr_gnb, 4))


print("---------------------------------------------------------------------------------------------------------------")
# -----------------------------------------------------LOGISTIC_REGRESSION----------------------------------------------
model = LogisticRegression()
model.fit(X_train,y_train)
prediction6=model.predict(X_test)

print("Logistic Regression AC score: ",round(metrics.accuracy_score(prediction6, y_test), 4))

f1_lr = f1_score(y_test, prediction6, average='weighted')
print("Logistic Regression F1 score:", round(f1_lr, 4))

recall_lr = recall_score(y_test, prediction6, average='weighted')
print("Logistic Regression Recall score:", round(recall_lr, 4))

precision_lr = precision_score(y_test, prediction6)
print("Logistic Regression Precision score:", round(precision_lr, 4))

geo_mean_lr = geometric_mean(precision_lr, recall_lr)

print("Logistic Regression Geometric Mean score:", round(geo_mean_lr, 4))

y_scores_lr = model.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_scores_lr[:, 1])
auc_pr_lr = auc(recall, precision)
print("Logistic Regression AUC-PR score:", round(auc_pr_lr, 4))


print("---------------------------------------------------------------------------------------------------------------")