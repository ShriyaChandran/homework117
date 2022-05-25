import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
print(df.head())

y = df['class']
X = df[['variance','skewness','curtosis','entropy']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#X = np.reshape(variance_train.ravel(), (len(variance_train), 1))
#Y = np.reshape(type_train.ravel(), (len(type_train), 1))

#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X, Y)

#X_test = np.reshape(variance_test.ravel(), (len(variance_test), 1))
#Y_test = np.reshape(type_test.ravel(), (len(type_test), 1))

#type_prediction = classifier.predict(X_test)


LR = LogisticRegression()
LR.fit(X_train,y_train) #fitting the model 

y_prediction = LR.predict(X_test)



predicted_values = []
for i in y_prediction:
  if i == 0:
    predicted_values.append("Authorized")
  else:
    predicted_values.append("Forged")

actual_values = []
for i in y_test:
  if i == 0:
    actual_values.append("Authorized")
  else:
    actual_values.append("Forged")


labels = ["Forged", "Authorized"]

cm = confusion_matrix(actual_values, predicted_values)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
#fig.show()