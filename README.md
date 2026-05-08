#  ¿Por qué decidió eso el computador?
### Demo de IA Explicable (XAI) con SHAP

Repositorio de apoyo para la exposición sobre **Explainable AI (XAI)**.  
El código entrena un modelo de Machine Learning que predice riesgo cardiovascular y luego usa **SHAP** para explicar, variable por variable, por qué el modelo tomó cada decisión.

---

##  ¿Qué es XAI?

Los modelos de IA pueden tomar decisiones muy precisas, pero sin explicar el porqué. Esto es un problema en áreas críticas como medicina, créditos o justicia.

**XAI (Explainable AI)** son técnicas que nos permiten abrir esa "caja negra" y entender qué factores influyeron en cada predicción.

```
Sin XAI:  Datos →  Modelo → Decisión        (¿por qué? no sé.)
Con XAI:  Datos →  Modelo → Decisión + Explicación 
```

---

##  ¿Qué hace este demo?

| Paso | Descripción |
|------|-------------|
| 1️ | Genera 500 pacientes ficticios con variables médicas reales |
| 2️  | Entrena un `RandomForestClassifier` para predecir riesgo cardiovascular |
| 3️  | Muestra el problema: el modelo decide pero **no explica por qué** |
| 4️  | Aplica **SHAP** para calcular el impacto de cada variable |
| 5️  | Imprime la explicación detallada por paciente en consola |
| 6️  | Genera 3 gráficas explicativas y las guarda como `.png` |

---

##  Cómo ejecutarlo

**1. Clona el repositorio:**
```bash
git clone https://github.com/tu-usuario/nombre-del-repo.git
cd nombre-del-repo
```

**2. Instala las dependencias:**
```bash
pip install shap scikit-learn pandas matplotlib
```

**3. Ejecuta el script:**
```bash
python xai_demo.py
```

---

##  Archivos

```
 repositorio
 ┣ xai_demo.py               ← código principal
 ┣ README.md                 ← este archivo
 ┣  grafica_1_importancia_global.png   ← se genera al correr el script
 ┣  grafica_2_waterfall_carlos.png     ← se genera al correr el script
 ┗  grafica_3_comparacion.png          ← se genera al correr el script
```

---

##  ¿Cómo funciona SHAP?

SHAP (**SH**apley **A**dditive ex**P**lanations) asigna a cada variable un valor numérico que indica **cuánto empujó** la predicción hacia arriba o hacia abajo.

```
Predicción final = Valor base + Σ(valores SHAP de cada variable)
```

Ejemplo de salida del demo:

```
┌─ Carlos (65 años, fuma)
│  Decisión: RIESGO ALTO  (100.0%)
│  Base del modelo: 77.1%
│
│  Factores que AUMENTARON el riesgo (+):
│    ↑ Presión sistólica   = 160    +5.5%  █████
│    ↑ Edad                = 65     +5.1%  █████
│    ↑ Diabetes            = Sí     +3.8%  ███
│    ↑ Tabaquismo          = Sí     +2.4%  ██
│
│  Factores que REDUJERON el riesgo (-):
│    (ninguno en este caso)
└─────────────────────────────────────────
```

---

## Limitaciones de XAI

- **Explicar ≠ tener razón** — el modelo puede dar una explicación coherente y aun así estar equivocado.
- **Se puede hacer trampa** — es posible construir un modelo injusto que muestre explicaciones que parecen correctas.
- **Las respuestas pueden variar** — los valores SHAP no son una verdad absoluta.
- **Puede ser lento** — calcular SHAP en modelos grandes con muchos datos toma tiempo.

---

## Tecnologías usadas

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-XAI-brightgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualización-red)

---

## Referencias

- [Documentación oficial de SHAP](https://shap.readthedocs.io/)
- [scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
