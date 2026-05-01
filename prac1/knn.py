from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from main import load_data

knn_model = KNeighborsClassifier(n_neighbors=3)

X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

#print(y_pred)

print(classification_report(y_test, y_pred))