# код из юпитер-ноутбука

import pandas as pd
from clearml import Task
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# импортируем датасет
wine_data = pd.read_csv('winequality-red.csv')

# делим на фичи и таргет
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# делим на трейн и на тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# инициализирую clearml
task = Task.init(project_name='Final Project', task_name='Model Training', task_type=Task.TaskTypes.optimizer)


# подбираем гиперпараметры и обучаем линейную регрессию (ридж)
linear_model_ridge = Ridge()

linear_param_grid = {
    'alpha': [0.1, 0.5, 1],
    'fit_intercept': [True, False]
}

linear_grid_search = GridSearchCV(linear_model_ridge, linear_param_grid, cv=5, scoring='neg_mean_squared_error')
linear_grid_search.fit(X_train, y_train)

best_linear_model = linear_grid_search.best_estimator_
linear_predictions = best_linear_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, linear_predictions)
mse_linear = mean_squared_error(y_test, linear_predictions)
r2_linear = r2_score(y_test, linear_predictions)

# логирую результаты
for param, value in linear_grid_search.best_params_.items():
    task.get_logger().report_scalar(param, "Ridge", value=value, iteration=0)

task.get_logger().report_scalar("MAE", "Ridge", value=mae_linear, iteration=0)
task.get_logger().report_scalar("MSE", "Ridge", value=mse_linear, iteration=0)
task.get_logger().report_scalar("R²", "Ridge", value=r2_linear, iteration=0)


print("Лучшие гиперпараметры для линейной регрессии:", linear_grid_search.best_params_)
print("Метрики линейной регрессии:")
print(f"MAE: {mae_linear}, MSE: {mse_linear}, R²: {r2_linear}")


# то же самое, но с лассо вместо риджа
linear_model_lasso = Lasso()

linear_param_grid = {
    'alpha': [0.1, 0.5, 1],
    'fit_intercept': [True, False]
}

linear_grid_search = GridSearchCV(linear_model_lasso, linear_param_grid, cv=5, scoring='neg_mean_squared_error')
linear_grid_search.fit(X_train, y_train)

best_linear_model = linear_grid_search.best_estimator_
linear_predictions = best_linear_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, linear_predictions)
mse_linear = mean_squared_error(y_test, linear_predictions)
r2_linear = r2_score(y_test, linear_predictions)

# логирую результаты
for param, value in linear_grid_search.best_params_.items():
    task.get_logger().report_scalar(param, "Lasso", value=value, iteration=0)

task.get_logger().report_scalar("MAE", "Lasso", value=mae_linear, iteration=0)
task.get_logger().report_scalar("MSE", "Lasso", value=mse_linear, iteration=0)
task.get_logger().report_scalar("R²", "Lasso", value=r2_linear, iteration=0)


print("Лучшие гиперпараметры для линейной регрессии:", linear_grid_search.best_params_)
print("Метрики линейной регрессии:")
print(f"MAE: {mae_linear}, MSE: {mse_linear}, R²: {r2_linear}")


# подбираем гиперпараметры и обучаем случайный лес
random_forest_model = RandomForestRegressor(random_state=42)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid_search = GridSearchCV(random_forest_model, rf_param_grid, cv=5, scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)

best_rf_model = rf_grid_search.best_estimator_
rf_predictions = best_rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, rf_predictions)
mse_rf = mean_squared_error(y_test, rf_predictions)
r2_rf = r2_score(y_test, rf_predictions)

# логирую результаты
for param, value in rf_grid_search.best_params_.items():
    task.get_logger().report_scalar(param, "Random Forest", value=value, iteration=0)

task.get_logger().report_scalar("MAE", "Random Forest", value=mae_rf, iteration=0)
task.get_logger().report_scalar("MSE", "Random Forest", value=mse_rf, iteration=0)
task.get_logger().report_scalar("R²", "Random Forest", value=r2_rf, iteration=0)

print("Лучшие гиперпараметры для случайного леса:", rf_grid_search.best_params_)
print("Метрики случайного леса:")
print(f"MAE: {mae_rf}, MSE: {mse_rf}, R²: {r2_rf}")

# закрываем таск clearml
task.close()


# случайный лес оказался лучше по всем метрикам: MAE и MSE меньше, а R² больше

# обучаем лучшую из трех моделей на лучших параметрах
random_forest_model = RandomForestRegressor(random_state=42,
                                            n_estimators=200,
                                            max_depth=20,
                                            min_samples_split=2,
                                            min_samples_leaf=1)


random_forest_model.fit(X_train, y_train)

# предсказания
rf_predictions = random_forest_model.predict(X_test)

# вычисляю метрики
mae_rf = mean_absolute_error(y_test, rf_predictions)
mse_rf = mean_squared_error(y_test, rf_predictions)
r2_rf = r2_score(y_test, rf_predictions)

# логирую результаты
print("Метрики случайного леса:")
print(f"MAE: {mae_rf}, MSE: {mse_rf}, R²: {r2_rf}")

print("Обучили лучшую модель на лучших параметрах")
