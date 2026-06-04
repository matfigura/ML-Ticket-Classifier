# ML Ticket Classifier

Projekt uczenia maszynowego do automatycznej klasyfikacji zgłoszeń supportowych.

System analizuje treść zgłoszenia i przewiduje:

* **priorytet zgłoszenia** — `low`, `medium`, `high`,
* **kolejkę obsługi** — dział, do którego powinno trafić zgłoszenie.

Celem projektu jest zasymulowanie prostego systemu ticket triage, który może wspierać helpdesk w kierowaniu zgłoszeń do odpowiedniego zespołu z odpowiednim priorytetem.

## Technologie

* Python
* Pandas
* NumPy
* Scikit-learn
* SQLite
* FastAPI
* Docker
* GitLab CI/CD
* Pytest
* Ruff
* Joblib
* Matplotlib

## Dataset

W projekcie wykorzystano dataset `Tobi-Bueck/customer-support-tickets` z Hugging Face.

W pierwszej wersji projektu użyto wyłącznie zgłoszeń w języku angielskim.

Po przygotowaniu danych zbiór zawiera:

* **28 261 zgłoszeń**
* **3 klasy priorytetu**
* **10 kolejek obsługi**
* **4 typy zgłoszeń**

## Funkcjonalności

* pobieranie datasetu,
* zapis danych do bazy SQLite,
* podstawowa eksploracyjna analiza danych,
* trenowanie modeli klasyfikacji tekstu,
* porównanie kilku modeli ML,
* zapis finalnych modeli do plików `.joblib`,
* predykcja priorytetu i kolejki dla nowych zgłoszeń,
* zapis predykcji do SQLite,
* REST API w FastAPI,
* uruchamianie aplikacji w Dockerze,
* pipeline CI/CD w GitLabie.

## Workflow projektu

1. Pobranie datasetu
2. Załadowanie danych do SQLite
3. Eksploracyjna analiza danych
4. Trenowanie modeli bazowych
5. Porównanie modeli
6. Zapis finalnych modeli
7. Predykcja nowych zgłoszeń
8. Udostępnienie modelu przez REST API
9. Konteneryzacja z użyciem Dockera
10. Automatyczne testy i CI/CD w GitLabie

## Modele

Tekst zgłoszenia jest przekształcany do reprezentacji liczbowej za pomocą **TF-IDF**. Następnie klasyfikacja odbywa się przy użyciu modeli ze Scikit-learn.

Porównane modele:

* Multinomial Naive Bayes
* Logistic Regression
* LinearSVC

Najlepsze wyniki dla obu zadań uzyskał **LinearSVC**.

## Wyniki

| Target    | Najlepszy model | Accuracy | Macro F1 |
| --------- | --------------: | -------: | -------: |
| Priorytet |       LinearSVC |   0.6736 |   0.6630 |
| Kolejka   |       LinearSVC |   0.6467 |   0.6589 |

Predykcja kolejki jest trudniejszym zadaniem, ponieważ obejmuje 10 klas. Model dla kolejki znacząco przewyższył prosty baseline większościowy, który osiągałby około 28,8% accuracy.

## Przykładowa predykcja

Dane wejściowe:

```text
Subject: Product return request
Body: I received a damaged product and would like to return it or exchange it.
```

Wynik modelu:

```text
Predicted priority: low
Predicted queue: returns and exchanges
```

Drugi przykład:

```text
Subject: Invoice issue
Body: I was charged twice for my last invoice and need help with the payment.
```

Wynik modelu:

```text
Predicted priority: high
Predicted queue: billing and payments
```

## REST API

Projekt udostępnia model przez REST API zbudowane w FastAPI.

Dostępne endpointy:

| Endpoint                          | Opis                                                 |
| --------------------------------- | ---------------------------------------------------- |
| `GET /health`                     | sprawdzenie statusu API i dostępności modeli         |
| `POST /predict`                   | predykcja priorytetu i kolejki dla nowego zgłoszenia |
| `GET /sample-tickets`             | lista przykładowych zgłoszeń                         |
| `GET /sample-tickets/{ticket_id}` | pobranie jednego przykładowego zgłoszenia            |
| `GET /predictions`                | ostatnie predykcje zapisane w SQLite                 |

Przykładowe zapytanie do `/predict`:

```json
{
  "subject": "Invoice issue",
  "body": "I was charged twice for my last invoice and need help with the payment.",
  "save_to_database": true
}
```

Przykładowa odpowiedź:

```json
{
  "predicted_priority": "high",
  "predicted_queue": "billing and payments",
  "model_version": "tfidf_linearsvc_v1"
}
```

Interaktywna dokumentacja API jest dostępna pod adresem:

```text
http://127.0.0.1:8000/docs
```

## Struktura projektu

```text
ml-ticket-classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── database/
│
├── models/
│   ├── priority_model.joblib
│   └── queue_model.joblib
│
├── reports/
│
├── src/
│   ├── api.py
│   ├── basic_eda.py
│   ├── check_database.py
│   ├── check_predictions.py
│   ├── compare_text_models.py
│   ├── config.py
│   ├── download_dataset.py
│   ├── export_tickets_sample.py
│   ├── load_data_to_sql.py
│   ├── predict.py
│   └── train_final_models.py
│
├── tests/
│
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── .gitlab-ci.yml
├── requirements.txt
└── README.md
```

## Uruchomienie lokalne

Utworzenie środowiska wirtualnego:

```bash
python -m venv .venv
```

Aktywacja środowiska na Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Instalacja zależności:

```bash
pip install -r requirements.txt
```

Pobranie datasetu:

```bash
python -m src.download_dataset
```

Załadowanie danych do SQLite:

```bash
python -m src.load_data_to_sql
```

Sprawdzenie bazy danych:

```bash
python -m src.check_database
```

Uruchomienie EDA:

```bash
python -m src.basic_eda
```

Porównanie modeli:

```bash
python -m src.compare_text_models --target all
```

Trenowanie finalnych modeli:

```bash
python -m src.train_final_models
```

Uruchomienie predykcji z terminala:

```bash
python -m src.predict
```

Uruchomienie REST API lokalnie:

```bash
uvicorn src.api:app --reload
```

## Uruchomienie przez Docker

Zbudowanie obrazu:

```bash
docker compose build
```

Uruchomienie API:

```bash
docker compose up api
```

API będzie dostępne pod adresem:

```text
http://127.0.0.1:8000/docs
```

Zatrzymanie kontenera:

```bash
docker compose down
```

## Testy

Uruchomienie testów:

```bash
pytest
```

Testy obejmują podstawową weryfikację endpointów API, m.in.:

* `/health`,
* `/predict`,
* walidację błędnego requestu do `/predict`.

## CI/CD

W projekcie skonfigurowano pipeline CI/CD z wykorzystaniem GitLab CI.

Pipeline uruchamia się automatycznie po wypchnięciu zmian do repozytorium i składa się z etapów:

* `lint` — sprawdzenie jakości kodu za pomocą Ruff,
* `format_check` — sprawdzenie formatowania kodu,
* `tests` — uruchomienie testów automatycznych za pomocą pytest,
* `docker_build` — sprawdzenie, czy aplikacja poprawnie buduje się jako obraz Docker.

Dzięki temu pipeline pozwala wykryć problemy związane z jakością kodu, testami API oraz konfiguracją konteneryzacji.

## Ograniczenia

* Modele bazują wyłącznie na treści zgłoszenia.
* Priorytet może zależeć od dodatkowego kontekstu biznesowego, np. SLA, typu klienta lub liczby użytkowników dotkniętych problemem.
* Niektóre kolejki są do siebie znaczeniowo podobne, np. `technical support`, `it support`, `product support` i `customer service`.
* Wyniki decyzyjne modelu LinearSVC nie są prawdopodobieństwami.

## Możliwe usprawnienia

* strojenie hiperparametrów z użyciem GridSearchCV,
* dodanie progu pewności i ręcznej weryfikacji niepewnych predykcji,
* rozbudowanie testów jednostkowych i integracyjnych,
* dodanie prostego interfejsu webowego,
* wdrożenie aplikacji w chmurze,
* monitorowanie jakości predykcji w czasie.
