import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=np.load("olivetti_faces.npy")
target=np.load("olivetti_faces_target.npy") #loading the data


print("There are {} images in the dataset".format(len(data)))
print("There are {} unique targets in the dataset".format(len(np.unique(target))))
print("Size of each image is {}x{}".format(data.shape[1],data.shape[2]))
print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4])) #analysis of the data

print("unique target number:",np.unique(target))

X=data.reshape((data.shape[0],data.shape[1]*data.shape[2])) #reshape array
print("X shape:",X.shape)

X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)
print("y_test:")
print(y_test )

print("X_test:")
print(X_test )

#print("X_train shape:",X_train.shape)
#print("y_train shape:{}".format(y_train.shape))



#plt.figure(1, figsize=(12, 8))

#plt.plot(pca.explained_variance_, linewidth=2)

#plt.xlabel('Components')
#plt.ylabel('Explained Variaces')
#plt.show()

n_components=50
pca=PCA(n_components=n_components, whiten=True)  #transforms a random vector into a white noise vector with uncorrelated components.
pca.fit(X_train)# fit the model with X

X_train_pca=pca.transform(X_train) #Apply dimensionality reduction to X.
X_test_pca=pca.transform(X_test)


models=[]
models.append(("LogReg",LogisticRegression()))

for name, model in models:
    clf = model  #one by one taking up the model from logistic models

    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    print(10 * "=", "{} Result".format(name).upper(), 10 * "=")
    print("Accuracy score:{:0.2f}".format(metrics.accuracy_score(y_test, y_pred)))
    print()

print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


