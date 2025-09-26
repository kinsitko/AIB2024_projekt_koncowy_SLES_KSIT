# Analiza danych i model predykcyjny wyników uczniów w matematyce

Projekt końcowy w ramach studiów *AI i automatyzacja procesów biznesowych, edycja 2024*, Politechnika Gdańska.  
Celem jest eksploracja danych dotyczących uczniów oraz budowa modelu regresyjnego przewidującego wyniki końcowe na podstawie cech demograficznych, społecznych i edukacyjnych.

##  Struktura repozytorium

projekt_koncowy_AIB2024_KSIT_SLES/
│
├── data/ # dane
│
├── notebooks/
│ └── EDA.ipynb # analiza eksploracyjna danych
│
├── src/
│ ├── config/ # pliki konfiguracyjne
│ ├── processing/ # trenowanie, tuning i ewaluacja modeli, pipelin'y
│ └── utils/ # funkcje pomocnicze i niezbędne importy
│
├── requirements.txt # lista wymaganych bibliotek
├── README.md # opis projektu
└── .gitignore # ignorowane pliki


## Dane

- Dataset: [Math Students – Kaggle](https://www.kaggle.com/datasets/adilshamim8/math-students)  
- Format: CSV (`Math-Students.csv`)  
- Dane zawierają cechy dotyczące uczniów (m.in. czas nauki, absencje, alkohol, edukacja rodziców, płeć, itp.) oraz oceny cząstkowe i końcowe.  

Dane **nie są przechowywane w repozytorium** (`data/` jest w `.gitignore`).  
Aby pobrać dane:
```bash
kaggle datasets download -d adilshamim8/math-students -p data/raw --unzip

```

## Wymagania

Do uruchomienia projektu potrzebny jest Python 3.10+ oraz pakiety z pliku requirements.txt.

## Uruchamianie

1. Analiza danych (EDA)
Uruchom notebook:
jupyter notebook notebooks/EDA.ipynb


2. Trenowanie modelu i predykcja danych
Pipeline treningowy i użycie wytrenowanego modelu na nowych danych:
python src/processing/math_sles_ksit(najnowsza_data).py

## Wyniki

- Modele testowane: 
    - Główny model XGBoost,
    - Modele porównawcze: Linear Regression, Lasso Regression, RandomForest, KNN Regression
- Optymalizacja hiperparametrów: Optuna
- Metryki ewaluacyjne: MSE, MAE, R²
- Wizualizacje: rozkłady cech, korelacje, SHAP feature importance


**Wybrane cechy feature importance:**
['num__failures' 'num__goout' 'num__absences' 'cat__sex_M'
 'cat__Pstatus_T' 'cat__Fjob_other' 'cat__Fjob_teacher'
 'cat__guardian_other' 'cat__schoolsup_yes' 'cat__higher_yes']

 **Najlepsze hiperparametry wg Optuna:**
 {'n_estimators': 175, 'max_depth': 3, 'learning_rate': 0.04279589903340042, 'subsample': 0.5551067007705357, 'colsample_bytree': 0.7485072229441385}



**Porównanie wyników z innymi modelami (train/test split, bez CV):**
                Model        MSE       MAE        R2
0      XGB Regressor  19.131135  3.366837  0.073938
1  Linear Regression  19.787851  3.424421  0.042149
2   Lasso Regression  20.337674  3.479875  0.015534
3      Random Forest  23.799324  3.638400 -0.152030
4     KNN Regression  21.705500  3.717500 -0.050677

**Wyniki dla XGB - z train/test split:**

MSE: 19.131135
MAE: 3.366836
R²: 0.073938

**Wyniki dla XGB - z 3-fold CV:**
      
Metric       mean     median       std
MSE     15.281486  16.825067  2.736966
MAE      2.988295   3.020380  0.243937
R²       0.268575   0.230307  0.110279

## Technologie

Python
scikit-learn
XGBoost
Optuna
Pandas, NumPy
Matplotlib, Seaborn
SHAP
Jupyter Notebook

## Autorzy

- Sylwia Lesner (GitHub @kondratowicz95)
- Kinga Sitkowska (GitHub @kinsitko)