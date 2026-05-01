from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from main import load_data

nb_model = GaussianNB()

X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

nb_model = nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

print(classification_report(y_test, y_pred))