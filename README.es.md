[English](README.md) | **Español**

# Titanic: por qué una regla de dos líneas le gana a siete modelos ajustados en el leaderboard

![CI](https://github.com/Rxyxs/titanic-survival-kaggle/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![License](https://img.shields.io/badge/license-MIT-green)

La competencia introductoria de Kaggle
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic):
891 pasajeros con resultado conocido, 418 por predecir, evaluados por accuracy.

La recorrí completa, como pide la competencia: EDA, ingeniería de variables, siete familias
de modelos con validación cruzada anidada, dos ensembles y envíos. Después me hice la
pregunta que el leaderboard obliga a hacer: **¿por qué modelos que marcan 84% en validación
cruzada caen a 77% en el test, mientras una regla simple se sostiene?**

**Mejor puntaje en Kaggle: 0.79904** (334 de 418 pasajeros), con la regla mujer-niño
agrupada por familia. El mejor envío de machine learning sacó 0.77511.

**Página del proyecto:** https://rxyxs.github.io/titanic-survival-kaggle/

## Resultados

La validación cruzada es anidada (5 folds externos x 3 repeticiones, grid search interno de
5 folds), así que el ajuste de hiperparámetros nunca ve el fold con que se evalúa. Las reglas
usan 5 folds x 10 repeticiones. Kaggle evalúa cada envío sobre los 418 pasajeros del test
(cada puntaje es múltiplo de 1/418), así que **un pasajero son 0.24 puntos**.

| Envío | Accuracy CV | Kaggle |
|---|---|---|
| Regla de género (mujeres sobreviven, hombres no) | 0.787 | 0.76555 |
| **Regla mujer-niño, grupos por familia** (elegida por CV, ver hallazgo 5) | 0.840 | **0.79904** |
| Regla mujer-niño, grupos por pasaje | 0.837 | 0.79665 |
| Ensemble, voto suave de 7 modelos + variables de grupo | 0.846 | 0.77511 |
| Regresión logística + variables de grupo | 0.841 | 0.77272 |
| Ensemble, voto suave de 7 modelos, sin variables de grupo | 0.835 | 0.76794 |
| XGBoost sin variables de grupo (mejor modelo individual sin grupos) | 0.837 | 0.76076 |
| LightGBM con una variable de grupo con fuga | 0.986 | 0.64114 |

![CV vs Kaggle](docs/img/validation.png)

**1. Las apuestas de cada modelo contra la regla de género se vuelven una moneda al aire en
el test.** El sexo explica la mayor parte del resultado (sobrevivió el 74% de las mujeres y el 19% de los
hombres). En la práctica, un modelo es la regla de género más unas decenas de *apuestas*:
un hombre que predice que sobrevive o una mujer que predice que muere. Como la regla de
género también se envió, el valor de esas apuestas en el test sale de los dos puntajes, sin
etiquetas del test: `apuestas netas correctas = (puntaje - puntaje género) x 418`.

| | Apuestas en CV (por 418) | Aciertos en CV | Apuestas en test | Aciertos en test |
|---|---|---|---|---|
| Regresión logística + grupos | 40 | 78% | 33 | 55% (18 de 33) |
| Ensemble + grupos | 40 | 79% | 32 | 56% (18 de 32) |
| Regla mujer-niño, grupos por pasaje | 22 | 97% | 15 | **93% (14 de 15)** |
| Regla mujer-niño, grupos por familia | 23 | 98% | 16 | **94% (15 de 16)** |

![Apuestas](docs/img/bets.png)

Las apuestas de los modelos salen de patrones (hombres de primera clase, familias grandes de
tercera) que calzan con los 891 pasajeros de entrenamiento y no se trasladan. La regla solo
apuesta cuando hay evidencia directa: una mujer cuyos compañeros de pasaje (mujeres y niños)
murieron todos, o un niño cuyos compañeros de pasaje (mujeres y niños) sobrevivieron todos.
Esas apuestas mantienen su acierto. Con 32 apuestas la precisión en test es incierta
(aprox. +/-17 puntos al 95%), pero 78% queda fuera de ese rango, y el 14 de 15 de la regla no.

**2. La brecha entre CV y leaderboard no es solo sobreajuste.** La regla de género no ajusta
nada y aun así baja dos puntos (0.787 a 0.766), un error estándar con 418 pasajeros. Los
modelos bajan seis a siete puntos, así que unos cuatro puntos son apuestas que no
generalizan. El optimismo del ajuste de hiperparámetros es chico en comparación: el puntaje
del grid search interno supera al externo en a lo más 1.0 punto (0.5 en promedio).

**3. La supervivencia del grupo de pasaje es la única variable que rinde en el leaderboard,
y se filtra si se calcula sin cuidado.** 596 de los 1.309 pasajeros viajan en uno de 216
pasajes compartidos, y 152 pasajeros del test comparten pasaje con alguien del
entrenamiento. La variable de cada pasajero es la tasa de supervivencia de sus compañeros de
pasaje. Qué etiquetas puede usar decide lo que reporta la CV:

![Fuga](docs/img/leak.png)

| Etiquetas que puede usar la variable de grupo | CV regresión logística | CV LightGBM |
|---|---|---|
| Solo el fold de entrenamiento (lo que tendrá el test) | 0.841 | 0.838 |
| Todo el train excepto la etiqueta propia | 0.843 | 0.843 |
| Todo el train incluyendo la etiqueta propia | 0.989 | 0.986 |

Incluir la etiqueta propia es una fuga obvia en una revisión de código y catastrófica en la
práctica: para un pasajero que viaja solo, la variable *es* la respuesta. El modelo aprende a
confiar en ella, y en el test predice 57% de supervivencia y saca 0.641, peor que predecir
por sexo. La fuga sutil (etiquetas de compañeros en el fold de validación) infla la CV solo
0.2 a 0.5 puntos, y medirla es la única forma de saber que es chica.

**4. Con 891 pasajeros, elegir modelo es ruido.** La desviación estándar entre folds es de
unos 2 puntos, y un mismo modelo se mueve hasta 2.8 puntos entre repeticiones de la CV de
5 folds (XGBoost: 0.826 a 0.854). Las siete familias ajustadas con variables de grupo quedan
entre 0.829 y 0.841, y el ensemble (0.846) está a menos de una desviación estándar del mejor
modelo individual. En el leaderboard, el ensemble de voto suave le gana a la regresión
logística por exactamente un pasajero.

**5. Agrupar por familia en vez de por pasaje mejora un poco la regla, y la CV lo dice antes
que el leaderboard.** Un pasaje une a acompañantes que no son familia (niñeras, amigos) y
separa a familias que compraron pasajes distintos. Comparé tres agrupaciones sobre los mismos
100 folds (5 folds x 20 repeticiones):

| Grupo = | CV regla | vs pasaje | Repeticiones mejores | Precisión apuestas (CV) |
|---|---|---|---|---|
| Mismo pasaje | 0.836 | | | 97,3% |
| **Misma familia**: apellido + clase + pasaje sin último dígito + tarifa + puerto | **0.840** | +0.39 | 20 de 20 | 98,0% |
| Unidos por cualquiera de los dos (componentes conexas) | 0.840 | +0.33 | 20 de 20 | 97,5% |

La agrupación por familia se eligió solo por CV y se envió una vez. Cambia 3 de las 418
predicciones del test; Kaggle pasó de 0.79665 a 0.79904, un pasajero más acertado (16
apuestas en el test, 15 correctas). Coincide con la CV en la dirección, pero un pasajero está
dentro del ruido, así que la lectura honesta es "algo mejor, no demostradamente mejor".

## Enfoque

Los pasos de la propia competencia: aceptar reglas, descargar, entender, EDA, entrenar,
ajustar y combinar, enviar.

- **EDA** ([`reports/eda.json`](reports/eda.json)): supervivencia 38,4%; mujeres 74% vs
  hombres 19%; sobrevive el 97% de las mujeres de primera clase vs 50% en tercera; el título
  del nombre separa niños (Master, 57%) de hombres (Mr, 16%). Falta la edad en 263 de 1.309
  pasajeros y la cabina en 1.014.

![EDA](docs/img/eda.png)

- **Variables** ([`src/titanic/features.py`](src/titanic/features.py)): título, sexo,
  clase, edad imputada con la mediana de título x clase, indicador de edad faltante, tamaño
  de familia, indicador de viajar solo, pasajeros por pasaje, log de la tarifa y de la
  tarifa por persona, cubierta según la letra de la cabina, puerto. Usan solo atributos de
  los pasajeros y se calculan sobre train y test juntos. Las **variables de grupo de pasaje**
  usan etiquetas y se recalculan dentro de cada fold, solo con el fold de entrenamiento.
- **Modelos** ([`src/titanic/models.py`](src/titanic/models.py)): regresión logística, SVM
  RBF, k-NN, random forest, LightGBM, XGBoost y CatBoost, cada uno con una grilla chica; un
  ensemble de voto suave y otro de mayoría sobre los siete, con miembros fijados de antemano
  para evaluarlos con la misma CV anidada.
- **Regla mujer-niño**: la idea de grupos mujer-niño que circula en los notebooks públicos
  de la competencia, reducida a dos correcciones de la regla de género.
- **Análisis de apuestas** ([`src/titanic/bets.py`](src/titanic/bets.py)): cantidad y
  precisión de las apuestas en predicciones fuera de fold vs. las que implican los puntajes
  públicos.

No usé el leaderboard para elegir modelo: los primeros siete archivos salieron de una sola
corrida del pipeline antes de enviar nada, y se enviaron juntos. El octavo (grupos por
familia) se eligió por CV entre tres agrupaciones y se envió una sola vez.

## Datos

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic),
Kaggle (Will Cukierski, 2012). Descargados en octubre de 2026 con la API de Kaggle bajo las
reglas de la competencia (uso en la competencia, académico y no comercial); los datos no
están en este repositorio. `submissions/` tiene mis archivos de predicción.

## Reproducir

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m titanic.data       # descarga (requiere ~/.kaggle/kaggle.json y reglas aceptadas)
python -m titanic.run        # EDA, baselines, CV anidada, fugas, envíos (~5 min)
python -m titanic.submit gender woman_boy_rule xgboost_base ensemble_soft_base logreg_group ensemble_soft_group lightgbm_leaky
python -m titanic.rule_groups  # grupos por pasaje, familia o ambos para la regla; elige por CV
python -m titanic.submit woman_boy_rule_family
python -m titanic.bets       # apuestas: CV vs leaderboard
python -m titanic.figures '{\"gender\":\"gender\",\"rule_honest\":\"woman_boy_rule\",\"logreg\":\"logreg_group\",\"ensemble_soft\":\"ensemble_soft_group\"}'
pytest -q tests
```

## Estructura

```
src/titanic/  data.py      descarga y carga
              features.py  variables de pasajero, de grupo de pasaje, regla mujer-niño
              models.py    modelos, CV anidada, ajustes finales
              run.py       pipeline
              submit.py    envío con el CLI de Kaggle y puntajes
              rule_groups.py  definiciones de grupo para la regla, elegida por CV
              bets.py      precisión de las apuestas, CV vs leaderboard
              figures.py   figuras
reports/      cada número de este README (JSON)
submissions/  los ocho archivos enviados
tests/        tests de variables y de aislamiento entre folds
```

## Limitaciones

- Con 418 pasajeros en el test, cualquier diferencia de leaderboard bajo unos cuatro puntos
  es estadísticamente frágil; el análisis de apuestas habla de la dirección de la brecha, no
  de su tamaño exacto.
- Las grillas de hiperparámetros son chicas a propósito; no probé búsquedas más grandes
  porque la CV igual no logra ordenar los modelos.
- El puntaje público de Kaggle aquí cubre todo el test, así que no hay un puntaje privado
  separado con que contrastar.

## Licencia

Código: MIT. Datos: reglas de la competencia de Kaggle (no se redistribuyen).
