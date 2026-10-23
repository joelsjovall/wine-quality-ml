# Redovisningsguide – Wine Quality

Guiden följer notebooken från början till slut och förklarar vad koden gör, varför den behövs och hur resultaten ska beskrivas.

## Notebookcell 0 – Importer

```python
import pandas as pd
import matplotlib.pyplot as plt
```

Pandas används för tabeller och Matplotlib för histogram och ROC-kurvor.

**Säg:** “Jag importerar pandas för att hantera datasetet och Matplotlib för visualiseringar.”

## Notebookcell 1–2 – Datasetet

```python
df = pd.read_csv(url, sep=";")
df.head()
```

`url` är adressen till CSV-filen. `sep=";"` behövs eftersom semikolon skiljer kolumnerna åt. `df` är en DataFrame, alltså en tabell. `head()` visar de fem första raderna.

**Säg:** “Jag kontrollerar med `head()` att datasetet lästes in korrekt.”

## Notebookcell 3 – Kontrollera datan

```python
df.shape
df.columns
df.isnull().sum()
```

Datasetet har 1599 rader och 12 kolumner. Elva kolumner är kemiska egenskaper och `quality` är målvariabeln. Alla kolumner har noll saknade värden.

**Säg:** “Jag kontrollerar storlek, kolumner och saknade värden innan modellträningen.”

## Notebookcell 4 – Histogram

Histogrammet visar kvalitetsbetygens fördelning. X-axeln visar betyg och y-axeln antal viner. Få viner har betyg 7 eller högre.

**Koppling till särskild uppgift:** “Det gör klasserna obalanserade, så accuracy kan bli missvisande.”

## Notebookcell 6 – Features och målvariabel

```python
X = df.drop(columns=["quality"])
y = df["quality"]
```

`X` innehåller features, alltså egenskaper modellen använder. `y` är målvariabeln som modellen ska förutsäga. `quality` tas bort för att undvika dataläckage.

**Säg:** “Modellen får kemiska värden som input och ska förutsäga kvaliteten.”

## Notebookcell 6 – Träning, validering och test

Första uppdelningen sparar 20 procent till testdata. Den andra delar resten så att resultatet blir 60 procent träning, 20 procent validering och 20 procent test.

- Träningsdata: modellen lär sig.
- Valideringsdata: modeller och hyperparametrar jämförs.
- Testdata: används bara vid slututvärderingen.

`random_state=42` gör uppdelningen reproducerbar.

**Säg:** “Jag väljer modellen på valideringsdatan och sparar testdatan för ett rättvist slutprov.”

## Notebookcell 8 – Baseline för regression

```python
baseline_reg = DummyRegressor(strategy="mean")
```

DummyRegressor förutsäger alltid medelvärdet från träningsdatan. Den är en enkel jämförelsemodell.

Resultat: RMSE 0,802, MAE 0,686 och R² −0,014.

- RMSE: fel där stora fel straffas extra; lägre är bättre.
- MAE: genomsnittligt absolut fel; lägre är bättre.
- R²: hur mycket variation modellen förklarar; högre är bättre.

## Notebookcell 9 – Baseline för klassificering

```python
y_train_clf = (y_train >= 7).astype(int)
```

Kvalitet 7 eller högre blir klass 1, bra vin. Övriga blir klass 0.

```python
baseline_clf = DummyClassifier(strategy="most_frequent")
```

DummyClassifier väljer alltid den vanligaste klassen, alltså 0.

Resultat: accuracy 0,884, precision 0, recall 0, F1 0 och AUC 0,5.

- Precision: av de viner modellen kallar bra, hur många är bra?
- Recall: av alla bra viner, hur många hittar modellen?
- F1: kombination av precision och recall.
- AUC: hur bra modellen skiljer klasserna åt; 0,5 är slump.

## Notebookcell 10 – Extra klassificeringsmått för baseline

Detta block räknar ut precision, recall, F1, sannolikheter och AUC för DummyClassifier.

## Notebookcell 12 – LinearRegression och Pipeline

```python
linear_model = make_pipeline(StandardScaler(), LinearRegression())
```

LinearRegression försöker hitta linjära samband. StandardScaler gör features jämförbara i skala. Pipeline kopplar ihop scaling och modell.

**Säg:** “Scalern anpassas på träningsdata och används sedan på validerings- och testdata, så att jag undviker dataläckage.”

Resultat: RMSE 0,674, MAE 0,500 och R² 0,284.

## Notebookcell 14 – Ridge och alpha

Ridge är linjär regression med regularisering. Regularisering begränsar stora koefficienter och kan minska överanpassning.

`alpha` styr regulariseringens styrka. Jag testade 0,1, 1 och 10. Resultaten blev nästan identiska, så alpha hade liten effekt på detta dataset.

## Notebookcell 16 – RandomForestRegressor

Random Forest kombinerar flera beslutsträd. `max_depth` bestämmer hur djupa träden får bli.

| Djup | Träning RMSE | Validering RMSE |
|---:|---:|---:|
| 3 | 0,614 | 0,670 |
| 5 | 0,529 | 0,653 |
| 10 | 0,326 | 0,621 |

När djupet ökar kan modellen lära sig fler detaljer. Skillnaden mellan tränings- och valideringsfel ökar, vilket tyder på överanpassning. Jag valde ändå 10 eftersom valideringsfelet är lägst.

## Notebookcell 19 – Jämförelsetabell för regression

Tabellen samlar RMSE, MAE och R² för alla regressionsmodeller. Den gör modellvalet tydligt: RandomForestRegressor med depth 10 har lägst RMSE och högst R².

## Notebookcell 21–22 – LogisticRegression

LogisticRegression klassificerar vin som 0 eller 1 och kan ge sannolikheter. Den använder StandardScaler i en Pipeline.

Resultat: accuracy 0,887, precision 0,522, recall 0,324, F1 0,400 och AUC 0,851.

## Notebookcell 28 – Jämförelsetabell för klassificering

Tabellen samlar accuracy, precision, recall, F1 och AUC för klassificeringsmodellerna. Den visar varför RandomForestClassifier med depth 10 väljs.

## Notebookcell 24 – RandomForestClassifier

Jag testar `max_depth` 5 och 10. Med depth 10 fick modellen accuracy 0,919, precision 0,720, recall 0,486, F1 0,581 och AUC 0,882.

**Säg:** “Jag valde depth 10 eftersom den gav bäst resultat på valideringsdatan.”

## Notebookcell 26 – ROC-kurvor

ROC-kurvan visar resultatet vid olika trösklar.

- X-axel: False Positive Rate, dåliga viner som felaktigt kallas bra.
- Y-axel: True Positive Rate, samma som recall.
- Diagonalen: slumpens resultat.
- AUC 0,5: slump. AUC 1,0: perfekt.

## Notebookcell 30–31 – Särskild uppgift: accuracy och tröskel

```python
proba = rf10.predict_proba(X_val)[:, 1]
```

`predict_proba` hämtar sannolikheten för klass 1, alltså bra vin.

```python
pred = (proba >= threshold).astype(int)
```

Om sannolikheten är minst tröskeln blir förutsägelsen 1.

| Tröskel | Accuracy | Precision | Recall |
|---:|---:|---:|---:|
| 0,5 | 0,919 | 0,720 | 0,486 |
| 0,3 | 0,906 | 0,571 | 0,757 |
| 0,2 | 0,863 | 0,448 | 0,811 |

När tröskeln sänks hittar modellen fler bra viner. Recall ökar, men precision och accuracy minskar.

Jag väljer 0,3 eftersom vinhandlaren vill hitta många bra viner. Det är en kompromiss mellan recall 0,757 och precision 0,571.

## Notebookcell 33 – Slutlig testning på testdata

Testresultat:

- Regression: baseline RMSE 0,809, Random Forest RMSE 0,578.
- Klassificering: baseline accuracy 0,853, Random Forest accuracy 0,878.
- Random Forest: precision 0,643, recall 0,383, F1 0,480 och AUC 0,910.

**Säg:** “Testdatan användes först efter modellvalet. Därför är detta den slutliga kontrollen på ny data.”

## Om läraren ber om en ändring

- Högre `max_depth`: mer komplex modell, lägre träningsfel men större risk för överanpassning.
- Högre `alpha`: starkare regularisering; kan minska överanpassning men ge underanpassning om det blir för högt.
- Annan `random_state`: annan uppdelning och möjliga andra resultat.
- Lägre tröskel: fler positiva förutsägelser, högre recall men lägre precision.
- Ta bort `quality` från `X`: dataläckage eftersom modellen får se svaret.

## Muntlig grundstruktur

För varje kodblock: säg vad koden gör, varför den behövs och vad resultatet betyder.

“Jag gör detta för att ____. Resultatet blev ____. Det betyder att ____.”
