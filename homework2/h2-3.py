import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 read the data
df = pd.read_csv('diabetes.csv')

#2 check for null and print if 
print("null value: ")
print(df.isnull().sum())

#3 df describe 
print("\ndescriptive statistics:")
print(df.describe())

#4 num of rows with 0 values
columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nnumber of rows with 0 values: ")
for column in columns_with_zero:
    zero_count = (df[column] == 0).sum()
    print(f"{column}: {zero_count}")

#5 prime the data 
df[columns_with_zero] = df[columns_with_zero].replace(0, np.NaN)

df[columns_with_zero] = df[columns_with_zero].fillna(df[columns_with_zero].mean())

#6 normalize
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_scaled = scale(X)

#7 split train/test 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

#8 build 
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#9 compile with dif function 
model.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

#10 fit
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, batch_size=16, verbose=2, shuffle=True)

#11 train 
train_scores = model.evaluate(X_train, y_train, verbose=0)
print("\nTraining Accuracy: %.2f%%\n" % (train_scores[1]*100))
test_scores = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: %.2f%%\n" % (test_scores[1]*100))

#12 confusion matrix
y_test_pred_probs = model.predict(X_test)
y_test_pred = (y_test_pred_probs > 0.5).astype("int32")
c_matrix = confusion_matrix(y_test, y_test_pred)

# seaborn to plt 
sns.heatmap(c_matrix, annot=True, fmt='d',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            cbar=False, cmap='Blues')
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

# create ROC curve
FPR, TPR, thresholds = roc_curve(y_test, y_test_pred_probs)

# plot
plt.plot(FPR, TPR, label='ROC Curve')
plt.plot([0, 1], [0, 1], '--', color='black', label='Random Guess')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()



