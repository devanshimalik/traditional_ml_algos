from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from main import load_data

X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

lg_model = LogisticRegression()
lg_model= lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))