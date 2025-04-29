from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

joblib.dump(clf, "model.pkl")