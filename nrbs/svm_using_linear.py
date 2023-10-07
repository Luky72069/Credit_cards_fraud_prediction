import pandas as pd
from sklearn import svm #support vector Machine
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import recall_score, auc, precision_recall_curve, precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../creditcard.csv')

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
# -----------------------------------------------------LINEAR_SVM-------------------------------------------------------
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1, probability=True)
# Vytvorenie inštancie Triedy Borderline-SMOTE na oversampling
smote = BorderlineSMOTE(random_state=42)

# Vytvorenie inštancie Triedy TomekLinks na redukciu sumu
tomek = TomekLinks()

# Aplikácia redukcie sumu na trénovaciu množinu pred Borderline-SMOTE
X_train_cleaned, y_train_cleaned = tomek.fit_resample(X_train, y_train)

# Použitie Borderline-SMOTE na trénovaciu množinu bez šumu
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_cleaned, y_train_cleaned)

model.fit(X_train,y_train)

# Predikcia na testovacej množine
prediction1 = model.predict(X_test)

print("Linear SVM AC score: ",round(metrics.accuracy_score(prediction1, y_test), 4))

f1_svm = f1_score(y_test, prediction1, average='weighted')
print("Linear SVM F1 score:", round(f1_svm, 4))

recall_svm = recall_score(y_test, prediction1, average='weighted')
print("Linear SVM Recall score:", round(recall_svm, 4))

precision_svm = precision_score(y_test, prediction1)
print("Linear SVM Precision score:", round(precision_svm, 4))

geo_mean_svm = geometric_mean(precision_svm, recall_svm)

print("Linear SVM Geometric Mean score:", round(geo_mean_svm, 4))

y_scores_svm = model.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_scores_svm[:, 1])
auc_pr_svm = auc(recall, precision)
print("Linear SVM AUC-PR score:", round(auc_pr_svm, 4))

print("---------------------------------------------------------------------------------------------------------------")