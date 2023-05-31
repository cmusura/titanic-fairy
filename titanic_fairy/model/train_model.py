# """Modulo para entrenar y exportar un modelo de clasificacion para el problema de Titanic"""
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import StandardScaler


# X = strat_train_set.drop(["Survived"], axis=1)
# y = strat_train_set["Survived"]

# scaler = StandardScaler()
# X_data = scaler.fit_transform(X)
# y_data = y.to_numpy()


# clf = RandomForestClassifier()

# param_grid = {
#     "n_estimators": [10, 100, 200, 500],
#     "max_depth": [None, 5, 10],
#     "min_samples_split": [2, 3, 4],
# }

# grid_search = GridSearchCV(
#     clf, param_grid, cv=3, scoring="accuracy", return_train_score=True
# )
# grid_search.fit(X_data, y_data)
