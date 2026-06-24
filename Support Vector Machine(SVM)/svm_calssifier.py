import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

df = datasets.load_iris(as_frame=True).frame # type: ignore

df.head()

x = df.drop("target",axis=1)
y = df["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Model 
svc = SVC()
svc.fit(x_train_scaled,y_train)

y_pred = svc.predict(x_test_scaled)

print("Accuracy Score : ",accuracy_score(y_test,y_pred))
print("Classification Report : ",classification_report(y_test,y_pred))


# Linear Kernel

svc = SVC(kernel="linear")
svc.fit(x_train_scaled,y_train)

y_pred = svc.predict(x_test_scaled)

print("Accuracy Score : ",accuracy_score(y_test,y_pred))

# Polynomial Kernel

svc = SVC(kernel="poly")
svc.fit(x_train_scaled,y_train)

y_pred = svc.predict(x_test_scaled)

print("Accuracy Score : ",accuracy_score(y_test,y_pred))

# Sigmoid Kernel

svc = SVC(kernel="sigmoid")
svc.fit(x_train_scaled,y_train)

y_pred = svc.predict(x_test_scaled)

print("Accuracy Score : ",accuracy_score(y_test,y_pred))

# C_Values

c_vals = [0.5,1,2,3,4,5]
for c_val in c_vals:
    svc = SVC(C=c_val,kernel="rbf")
    svc.fit(x_train_scaled,y_train)

    y_pred = svc.predict(x_test_scaled)

    print("C = ",c_val,"Accuracy Score : ",accuracy_score(y_test,y_pred))
    



