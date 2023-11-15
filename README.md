Dokumentacja dla pliku Cwiczenia 2 perceptron.py

1. Wprowadzenie

Podany skrypt w języku Python implementuje klasyfikator perceptronowy wieloklasowy, wykorzystując bibliotekę scikit-learn i matplotlib do wizualizacji. Problem klasyfikacji wieloklasowej jest rozwiązany poprzez trenowanie wielu perceptronów binarnych, z których każdy odpowiada jednej klasie. 

2. Klasy

2.1. Klasa Perceptron
Atrybuty:
  - `eta` (float): Współczynnik uczenia.
  - `n_iter` (int): Liczba iteracji treningowych.
  - `w_` (tablica): Współczynniki wagowe.
  - `errors_` (lista): Lista do przechowywania liczby błędnych klasyfikacji w każdej epoce.

Metody:
  - `fit(X, y)`: Dopasuj perceptron do danych treningowych.
  - `net_input(X)`: Oblicz wejście sieci.
  - `predict(X)`: Przewiduj etykiety klas.

2.2. Klasa MultiClass
Atrybuty:
  - `perceptrons` (lista): Lista do przechowywania pojedynczych perceptronów dla każdej klasy.

Metody:
  - `__init__(train_data, train_label)`: Inicjalizuj instancję MultiClass i trenuj pojedyncze perceptrony dla każdej klasy.
  - `predict(X)`: Przewiduj etykiety klas dla danych wejściowych.

2.3. Funkcja Pomocnicza

Funkcja `num_of_unique_elements`:

Wejście:
  - `arr` (tablica): Tablica wejściowa.

Wyjście:
  - `int`: Liczba unikalnych elementów w tablicy wejściowej.

3. Główne Wykonanie

Funkcja `main`:

- Ładuje zbiór danych Iris, wydobywa cechy (`X`) i etykiety (`y`).
- Dzieli dane na zbiór treningowy i testowy.
- Tworzy instancję klasyfikatora `MultiClass` i trenuje go na danych treningowych.
- Dokonuje predykcji na danych testowych i wizualizuje obszary decyzyjne.

4. Uwaga

- Skrypt zakłada problem klasyfikacji binarnej dla każdego perceptronu w klasie `MultiClass`.

- Wizualizacja jest dostosowana do dwuwymiarowego zbioru danych.

- Skrypt używa zbioru danych Iris w celach demonstracyjnych. 

----------------------------------------

Dokumentacja dla pliku Cwiczenia 2 regresja liniowa.py

1. Wprowadzenie

Podany skrypt w języku Python implementuje klasyfikator regresji logistycznej wieloklasowej. Wykorzystuje on klasyfikator binarny regresji logistycznej do każdej klasy. Skrypt używa bibliotek NumPy, matplotlib, scikit-learn i mlxtend.

2. Klasy

2.1. Klasa Regresji Logistycznej (LogisticRegressionGD)

Atrybuty:
  - `eta` (float): Współczynnik uczenia.
  - `n_iter` (int): Liczba iteracji treningowych.
  - `random_state` (int): Ziarno generatora liczb losowych.

Metody:
  - `fit(X, y)`: Dopasuj model regresji logistycznej do danych treningowych.
  - `net_input(X)`: Oblicz wejście sieci.
  - `activation(z)`: Oblicz funkcję aktywacji.
  - `predict(X)`: Przewiduj etykiety klas.

2.2. Klasa MultiClassLogisticRegression

Atrybuty:
  - `eta` (float): Współczynnik uczenia.
  - `n_iter` (int): Liczba iteracji treningowych.
  - `random_state` (int): Ziarno generatora liczb losowych.
  - `classifiers` (lista): Lista zawierająca klasyfikatory dla każdej klasy.

Metody:
  - `fit(X, y)`: Dopasuj model regresji logistycznej wieloklasowej do danych treningowych.
  - `predict(X)`: Przewiduj etykiety klas.

3. Główne Funkcje

- Ładuje zbiór danych Iris, wydobywa cechy (`X`) i etykiety (`y`).
- Dzieli dane na zbiór treningowy i testowy.
- Standaryzuje dane treningowe i testowe.
- Tworzy instancję klasyfikatora `MultiClassLogisticRegression` i trenuje go na danych treningowych.
- Dokonuje predykcji na danych testowych i oblicza dokładność klasyfikacji.
- Wizualizuje obszary decyzyjne oraz prawdziwe etykiety klas.

4. Uwagi

- Klasyfikator binarny regresji logistycznej jest używany dla każdej klasy w celu implementacji klasyfikatora wieloklasowego.

- Wizualizacja obszarów decyzyjnych jest dostosowana do dwuwymiarowego zbioru danych.

- Skrypt korzysta z zestawu danych Iris w celach demonstracyjnych. 
