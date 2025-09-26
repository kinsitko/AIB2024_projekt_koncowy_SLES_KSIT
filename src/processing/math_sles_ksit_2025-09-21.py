# import sposób 1:
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# import sposób 2 i 1b:
from src.utils.imports import *   
# import sposób 3:
# from ..utils.imports import *


df = load_kaggle_dataset(
    dataset_name=config["dataset_name"],
    zip_file_name=config["zip_file_name"],
    csv_file_name=config["csv_file_name"],
    data_path=config["data_path"]
)

print(df.head())

target = 'G3'
df = df.drop(columns=['G1', 'G2'])

X = df.drop(columns=[target])
y = df[target]

# Podział na typy danych
categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])


def select_features(X, y):
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
    ])
    pipe.fit(X, y)
    model = pipe.named_steps['model']
    X_preprocessed = pipe.named_steps['preprocessor'].transform(X)
    selector = SelectFromModel(model, prefit=True)
    return selector, X_preprocessed, pipe

selector, X_transformed, full_pipe = select_features(X, y)
X_selected = selector.transform(X_transformed)

feature_names = full_pipe.named_steps['preprocessor'].get_feature_names_out()
selected_mask = selector.get_support()
selected_features = feature_names[selected_mask]

print("Wybrane cechy:")
print(selected_features)

# Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Optuna tuning
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }
    model = XGBRegressor(**params, random_state=42)
    return -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=3).mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# Finalny model
final_model = XGBRegressor(**study.best_params, random_state=42)
final_model.fit(X_train, y_train)

# Ewaluacja
y_pred = final_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Residual plot
sns.scatterplot(x=y_test, y=y_test - y_pred)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Rzeczywiste G3")
plt.ylabel("Błąd predykcji")
plt.title("Residual Plot")
plt.show()

# SHAP interpretacja

X_train_selected = pd.DataFrame(X_selected, columns=selected_features)
explainer = shap.Explainer(final_model, X_train_selected.astype(np.float64))
shap_values = explainer(X_train_selected.astype(np.float64), check_additivity=False)

shap.summary_plot(shap_values, X_train_selected, plot_type="bar")
shap.summary_plot(shap_values, X_train_selected)

# Zapis modelu
joblib.dump(final_model, 'xgb_g3_model.pkl')

# Porównanie z innymi modelami:

reg_models = {
    "XGB Regressor": final_model,
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(),
    "KNN Regression": KNeighborsRegressor()  
}
reg_results=[]

for name, model in reg_models.items():
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    reg_results.append([name, mse, mae, r2])
    # print(f"{name} -> MSE: {mse:.3f}, MAE:{mae:.3f}, R²:{r2:.3f}")
    
df_reg_results = pd.DataFrame(reg_results, columns=["Model", "MSE", "MAE", "R2"])
print("\nPorównanie wyników z innymi modelami (train/test split, bez CV):\n", df_reg_results)

reg_results_cv = {"Model": [], "Metric": [], "Value": []}
for name, model in reg_models.items():
    kf = KFold(
    n_splits=kfold_param["n_splits"],
    random_state=kfold_param["random_state"],
    shuffle=kfold_param["shuffle"])
    mse_scores = -cross_val_score(model, X_selected, y, scoring="neg_mean_squared_error", cv=kf)
    mae_scores = -cross_val_score(model, X_selected, y, scoring="neg_mean_absolute_error", cv=kf)
    r2_scores  = cross_val_score(model, X_selected, y, scoring="r2", cv=kf)

    for score in mse_scores:
        reg_results_cv["Model"].append(name)
        reg_results_cv["Metric"].append("MSE")
        reg_results_cv["Value"].append(score)
    for score in mae_scores:
        reg_results_cv["Model"].append(name)
        reg_results_cv["Metric"].append("MAE")
        reg_results_cv["Value"].append(score)
    for score in r2_scores:
        reg_results_cv["Model"].append(name)
        reg_results_cv["Metric"].append("R²")
        reg_results_cv["Value"].append(score)

df_reg_results_cv = pd.DataFrame(reg_results_cv)

xgb_cv = df_reg_results_cv[df_reg_results_cv["Model"] == "XGB Regressor"]
print("\nWyniki dla XGB (z CV):")
print(xgb_cv.groupby("Metric")["Value"].agg(["mean","median", "std"]))


plt.figure(figsize=(10,7))
sns.boxplot(data=df_reg_results_cv[df_reg_results_cv["Metric"]=="MSE"], x="Model", y="Value")
plt.xticks(rotation=20)
plt.title(f"Boxplot MSE ({kfold_param["n_splits"]}-fold CV)")
plt.show()

plt.figure(figsize=(10,7))
sns.boxplot(data=df_reg_results_cv[df_reg_results_cv["Metric"]=="MAE"], x="Model", y="Value")
plt.xticks(rotation=20)
plt.title(f"Boxplot MAE ({kfold_param["n_splits"]}-fold CV)")
plt.show()

plt.figure(figsize=(10,7))
sns.boxplot(data=df_reg_results_cv[df_reg_results_cv["Metric"]=="R²"], x="Model", y="Value")
plt.xticks(rotation=20)
plt.title(f"Boxplot R² ({kfold_param["n_splits"]}-fold CV)")
plt.show()