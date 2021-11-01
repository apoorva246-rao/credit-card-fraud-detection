import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')df
df.info()
df.insull().sum()
df.hist(figsize=(20,15),bins=30)
df['class '].value_counts()
df['Class'].value_counts(normalize=True)*100
df2 = df.loc[df['Class']==0].sample(n=492).copy()
df4 = pd.concat([df2,df3], ignore_indsex=True)
df4 = df4.sample(frac=1)
df4.reset_index(drop=True, inplace=True)df4
sns.displot(df4['Class'])
plt.figure(figsize=(20,10))sns.heatmap(df.corr(), annot=True, cmap='Accent')
sns.pairplot(df4[['V1', 'V2', 'V3', 'V4', 'V7', 'V9', 'V10', 'V11', 'V14', 'Class']], hue='Class')
df4.columns
X = df4[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]sel = VarianceThreshold(threshold=(.8))
sel.fit_transform(X)
X = sel.fit_transform(X)X[0]
y = df4['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.02, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rfc.fit(X_train, y_train)print('---Train Data---')
print(classification_report(y_train, y_pred=(rfc.predict(X_train))))
print("---Test Data---")
print(classification_report(y_test, y_pred=(rfc.predict(X_test))))
plot_confusion_matrix(rfc, X_test, y_test)
lr = LogisticRegression(max_iter=200) print('---Train Data---')
print(classification_report(y_train, y_pred=(lr.predict(X_train))))
print("---Test Data---")
print(classification_report(y_test, y_pred=(lr.predict(X_test))))
plot_confusion_matrix(lr, X_test, y_test)
 p1 = rfc.predict_proba(X_test)
p2 = lr.predict_proba(X_test)
 auc_score1 = roc_auc_score(y_test, p1[:,1])
auc_score2 = roc_auc_score(y_test, p2[:,1])
print(auc_score1,auc_score2)
 fpr1, tpr1, thresh1 = roc_curve(y_test, p1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, p2[:,1], pos_label=1)
 random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=plt.plot(fpr1, tpr1, linestyle='--', label='Random Forest')
plt.plot(fpr2, tpr2, linestyle='--', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')





