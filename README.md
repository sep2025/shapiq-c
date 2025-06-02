# shapiq-c
[SEP_SoSe2025 / Gruppe C] Game Theoretic Explainable Artificial Intelligence

Dieses Repository enthält die Projektarbeit im Rahmen des **Softwareentwicklungspraktikums SoSe 2025**.

## 📚 Projektbeschreibung
Im Projekt werden:
- neue Explainable AI (XAI) Methoden entwickelt und in shapiq integriert
- Unit-Tests, technische Dokumentationen erstellt

### 📌 Konkrete Aufgaben:
- **1. KNN-Shapley**: \
Implementierung eines neunen **KNNExplainer** innehalb der shapiq-Bibliothek.\
Dieser berechnet Shapley-Werte für KNN-Modelle (KNeighborsClassifier und KNeighborsRegressor) und ermöglicht so eine Erklärung dieser Vorhersagen die Modelle.\
Drei Varianten sollen implementiert werden:
  - KNN-Shapley (Jia et al. 2019)
  - Threshold KNN-Shapley (Wang et al. 2024)
  - Weighted KNN-Shapley (Wang et al. 2024)

- **2. Conditional Data Imputers**: \
Implementierung von zwei bedingten Imputationsmethoden (Aas et al. 2021, shapr Paper) zur Generierung fehlender Datenpunkte:
  - Gaussian Conditional Imputer: Bedingte Verteilung unter Annahme multivarater Normalverteilung
  - Gaussian Copula Conditional Imputer: Nutzung empirischer Ränder und einer gaußschen Kopula zur  Modellierung der Abhängigkeiten

- **3. Coalition-Finding Algorithmus**: \
Entwicklung eines heuristischen Algorithmuis zur effizienten Erkennung von Koalitionen in gewichteten Graphen aus Shapley-Interkationen.
Dabei sollen Knoten, Kanten und Hyper-Kanten berücksichtigt werden.\
Ziel ist es, die Koalitionen zu finden, die das vereinfachte Spiel maximieren oder minimieren.


## 📂 Projektstruktur und Ordner-Eigenschaften

| Ordner | Inhalt                                                 |
|--------|--------------------------------------------------------|
| `shapiq_student/` | Neue Funktionalitäten und Erweiterungen der Bibliothek |
| `test_grading/` | Vordefinierte Tests                                    |
| `tests/` | Eigene Unit-Tests für neue Funktionen                  |
| `docs/` | Erstellung und Pflege der Projektdokumentation         |
| `.github/` | GitHub Actions Workflows                               |


## 🚀 Projekt verwenden

### 1. Repository klonen
```bash
git clone https://github.com/sep2025/shapiq-c
cd shapiq-c
``` 


## 🔀 Branch-Regeln und Workflow

Dieses Repositrory verwendet zwei Haupt-Branches:

- **`main`**: 
  - **Haupt-Branch**, der den stabilen Stand des Projekts enthält
  - Direkte Pushes auf diesen Branch sind nicht erlaubt
  - Merge erfolgt nur über Pull-Requests (PRs) nach Review
  - **Mindestens 2 Reviews** sind erforderlich, bevor ein PR gemerged werden kann
  - Tags werden für Releases verwendet

- **`dev`**:
  - **Integrations- und Entwicklungs-Branch**
  - Hier erfolgt die Zusammenführung von Feature-Branches vor dem Merge in `main`
  - Direkte Pushes sind nicht erlaubt
  - PRs sind erforderlich, aber ohne Approval-Pflicht

💡 **Empfehlung**:
- Jeder sollte eingene Feature-Branches verwenden (z.B. `feature/knn-shapley`), um neue Funktionalitäten zu entwickeln und dann PRs in `dev` zu erstellen.
- Sobald `dev` stabil ist, kann ein PR in `main` über einen Release-Branch mit Tag (z.B. `release/v1.0.0`) erstellt werden.