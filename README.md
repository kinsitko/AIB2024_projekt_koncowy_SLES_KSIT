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

***Tu wkleić nasze wyniki*** 

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