import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('creditcard.csv')

def geometric_mean(precision_values, recall_values):
    return np.sqrt(np.multiply(precision_values, recall_values))

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

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

# List to append the score and then find the average
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []
geo_lst = []

# Inicializácia modelu LogisticRegression
lr_classifier = LogisticRegression()

# Definícia parametrov pre techniku SMOTE
smote = SMOTE(sampling_strategy='minority')

# Implementácia krížovej validácie s použitím SMOTE
for train, test in sss.split(original_Xtrain, original_ytrain):
    X_train_fold, y_train_fold = smote.fit_resample(original_Xtrain[train], original_ytrain[train])
    lr_classifier.fit(X_train_fold, y_train_fold)
    prediction = lr_classifier.predict(original_Xtrain[test])

    accuracy_lst.append(lr_classifier.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    geo_lst.append(geometric_mean(precision_score(original_ytrain[test], prediction), recall_score(original_ytrain[test], prediction)))



print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print("geometric mean: {}".format(np.mean(geo_lst)))
print('---' * 45)

labels = ['No Fraud', 'Fraud']
smote_prediction = lr_classifier.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))

y_score = lr_classifier.predict_proba(original_Xtest)[:, 1]

average_precision = average_precision_score(original_ytest, y_score)

print('Average precision-recall score: {0:0.4f}'.format(average_precision))
