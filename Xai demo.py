"""
=============================================================
  Demo: IA Explicable (XAI) con SHAP
  Tema: ¿Por qué decidió eso el computador?
=============================================================

¿Qué hace este código?
  Entrenamos un modelo de Machine Learning que predice
  el riesgo cardiovascular de un paciente.
  Luego usamos SHAP para explicar POR QUÉ el modelo
  tomó cada decisión, variable por variable.

Instalación:
  pip install shap scikit-learn pandas matplotlib

Cómo ejecutar:
  python xai_demo.py
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PASO 1: Crear datos sintéticos de pacientes
# ─────────────────────────────────────────────
# En un proyecto real, aquí cargarías tu CSV:
#   df = pd.read_csv("pacientes.csv")
#
# Nosotros generamos 500 pacientes ficticios
# para que el demo funcione sin archivos externos.

print("=" * 55)
print("  DEMO: IA Explicable (XAI) con SHAP")
print("  Predicción de riesgo cardiovascular")
print("=" * 55)
print()
print(" Paso 1: Generando datos de pacientes...")

np.random.seed(42)
N = 500

edad        = np.random.randint(25, 80, N)
presion     = np.random.randint(90, 200, N)
colesterol  = np.random.randint(120, 320, N)
imc         = np.round(np.random.uniform(17, 42, N), 1)
fuma        = np.random.randint(0, 2, N)       # 0=No, 1=Sí
ejercicio   = np.random.randint(0, 2, N)       # 0=No, 1=Sí
diabetes    = np.random.randint(0, 2, N)       # 0=No, 1=Sí
historial   = np.random.randint(0, 2, N)       # 0=Sin antecedentes, 1=Con antecedentes

# Etiqueta: riesgo alto (1) o bajo (0)
# La fórmula imita factores médicos reales
score = (
    (edad > 50).astype(int) * 2 +
    (presion > 140).astype(int) * 2 +
    (colesterol > 240).astype(int) +
    (imc > 30).astype(int) +
    fuma * 2 +
    (1 - ejercicio) +
    diabetes * 2 +
    historial
)
riesgo_alto = (score >= 5).astype(int)

df = pd.DataFrame({
    "edad":       edad,
    "presion":    presion,
    "colesterol": colesterol,
    "imc":        imc,
    "fuma":       fuma,
    "ejercicio":  ejercicio,
    "diabetes":   diabetes,
    "historial":  historial,
    "riesgo_alto": riesgo_alto
})

print(f"    {N} pacientes generados")
print(f"    Riesgo alto: {riesgo_alto.sum()} pacientes ({riesgo_alto.mean()*100:.0f}%)")
print(f"    Riesgo bajo: {(1-riesgo_alto).sum()} pacientes ({(1-riesgo_alto).mean()*100:.0f}%)")
print()

# ─────────────────────────────────────────────
# PASO 2: Entrenar el modelo
# ─────────────────────────────────────────────
print(" Paso 2: Entrenando el modelo RandomForest...")

FEATURES = ["edad", "presion", "colesterol", "imc",
            "fuma", "ejercicio", "diabetes", "historial"]
TARGET   = "riesgo_alto"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
modelo.fit(X_train, y_train)

precision = accuracy_score(y_test, modelo.predict(X_test))
print(f"    Modelo entrenado con {len(X_train)} pacientes")
print(f"    Precisión en datos de prueba: {precision*100:.1f}%")
print()

# ─────────────────────────────────────────────
# PASO 3: Predecir sin XAI  →  "caja negra"
# ─────────────────────────────────────────────
print(" Paso 3: Predicción SIN explicación (caja negra)...")

# Inventamos 3 pacientes nuevos para demostrar
pacientes_nuevos = pd.DataFrame({
    "edad":       [65,  35,  50],
    "presion":    [160, 115, 135],
    "colesterol": [260, 180, 210],
    "imc":        [32,  22,  27],
    "fuma":       [1,   0,   0],
    "ejercicio":  [0,   1,   1],
    "diabetes":   [1,   0,   0],
    "historial":  [1,   0,   1],
})

predicciones   = modelo.predict(pacientes_nuevos)
probabilidades = modelo.predict_proba(pacientes_nuevos)[:, 1]

nombres = ["Carlos (65 años, fuma)", "Sofía (35 años, activa)", "Luis (50 años, sedentario)"]

print()
print("   Paciente                      │ Riesgo  │ Probabilidad")
print("   ─────────────────────────────────────────────────────")
for nombre, pred, prob in zip(nombres, predicciones, probabilidades):
    etiqueta = "ALTO" if pred == 1 else " BAJO"
    print(f"   {nombre:<30} │ {etiqueta}  │ {prob*100:.1f}%")

print()
print("     El modelo decide pero NO explica por qué.")
print("      Aquí es donde entra XAI.")
print()

# ─────────────────────────────────────────────
# PASO 4: Aplicar SHAP  →  "caja transparente"
# ─────────────────────────────────────────────
print("Paso 4: Calculando valores SHAP (esto puede tardar unos segundos)...")

# TreeExplainer es la versión optimizada para
# modelos basados en árboles (RandomForest, XGBoost, etc.)
explainer   = shap.TreeExplainer(modelo)
shap_values = explainer.shap_values(pacientes_nuevos)

# Dependiendo de la versión de SHAP, el resultado puede venir
# como lista [clase_0, clase_1] o como array 3D (n, features, clases)
if isinstance(shap_values, list):
    sv = shap_values[1]            # versiones antiguas: lista de arrays
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]      # versiones nuevas: array 3D
else:
    sv = shap_values

print(f"    SHAP calculado para {len(pacientes_nuevos)} pacientes")
print()

# ─────────────────────────────────────────────
# PASO 5: Imprimir explicación por paciente
# ─────────────────────────────────────────────
print(" Paso 5: Explicación detallada por paciente")
print()

NOMBRES_VARIABLES = {
    "edad":       "Edad",
    "presion":    "Presión sistólica",
    "colesterol": "Colesterol",
    "imc":        "IMC (peso/talla²)",
    "fuma":       "Tabaquismo",
    "ejercicio":  "Ejercicio diario",
    "diabetes":   "Diabetes",
    "historial":  "Historial familiar",
}

for i, nombre in enumerate(nombres):
    pred  = predicciones[i]
    prob  = probabilidades[i]
    ev = explainer.expected_value
    valor_base = float(ev[1]) if hasattr(ev, '__len__') else float(ev)

    print(f"   ┌─ {nombre}")
    print(f"   │  Decisión: {' RIESGO ALTO' if pred==1 else ' RIESGO BAJO'}  ({prob*100:.1f}%)")
    print(f"   │  Base del modelo: {valor_base*100:.1f}% (promedio de todos los pacientes)")
    print(f"   │")
    print(f"   │  Factores que AUMENTARON el riesgo (+):")

    # Ordenar por impacto
    shap_i  = sv[i]
    pares   = sorted(zip(FEATURES, shap_i), key=lambda x: x[1], reverse=True)

    for feat, val in pares:
        valor_paciente = pacientes_nuevos.iloc[i][feat]
        if val > 0.005:
            barra = " " * min(int(abs(val) * 100), 20)
            if feat in ["fuma", "ejercicio", "diabetes", "historial"]:
                valor_str = "Sí" if valor_paciente == 1 else "No"
            else:
                valor_str = str(valor_paciente)
            print(f"   │    ↑ {NOMBRES_VARIABLES[feat]:<22} = {valor_str:<6}  +{val*100:.1f}%  {barra}")

    print(f"   │")
    print(f"   │  Factores que REDUJERON el riesgo (-):")
    for feat, val in reversed(pares):
        valor_paciente = pacientes_nuevos.iloc[i][feat]
        if val < -0.005:
            barra = " " * min(int(abs(val) * 100), 20)
            if feat in ["fuma", "ejercicio", "diabetes", "historial"]:
                valor_str = "Sí" if valor_paciente == 1 else "No"
            else:
                valor_str = str(valor_paciente)
            print(f"   │     {NOMBRES_VARIABLES[feat]:<22} = {valor_str:<6}  {val*100:.1f}%  {barra}")

    print(f"   └─────────────────────────────────────────────────")
    print()

# ─────────────────────────────────────────────
# PASO 6: Gráficas
# ─────────────────────────────────────────────
print(" Paso 6: Generando gráficas explicativas...")
print("   (Se abrirán 3 ventanas - ciérralas para continuar)")
print()

# ── Gráfica 1: Importancia global de variables ──
fig1, ax = plt.subplots(figsize=(9, 5))
fig1.patch.set_facecolor("#f8f8f8")
ax.set_facecolor("#f8f8f8")

sv_all      = explainer.shap_values(X_test)
if isinstance(sv_all, list):
    sv_all_pos = sv_all[1]
elif sv_all.ndim == 3:
    sv_all_pos = sv_all[:, :, 1]
else:
    sv_all_pos = sv_all
importancia = np.abs(sv_all_pos).mean(axis=0)
orden       = np.argsort(importancia)

colores = ["#5DCAA5" if v < importancia.max() * 0.5 else "#F0997B"
           for v in importancia[orden]]

bars = ax.barh(
    [NOMBRES_VARIABLES[FEATURES[i]] for i in orden],
    importancia[orden],
    color=colores, edgecolor="white", linewidth=0.5, height=0.6
)
ax.set_xlabel("Importancia SHAP promedio (impacto en la predicción)", fontsize=10)
ax.set_title("Importancia global de variables\n¿Qué factores influyen más en el modelo?",
             fontsize=12, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)

for bar, val in zip(bars, importancia[orden]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="#444")

leyenda = [
    mpatches.Patch(color="#F0997B", label="Alta importancia"),
    mpatches.Patch(color="#5DCAA5", label="Menor importancia"),
]
ax.legend(handles=leyenda, fontsize=9, framealpha=0.6)
plt.tight_layout()
plt.savefig("grafica_1_importancia_global.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Gráfica 2: Waterfall de Carlos (caso de riesgo alto) ──
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor("#f8f8f8")
ax2.set_facecolor("#f8f8f8")

paciente_idx = 0
shap_carlos  = sv[paciente_idx]
ev_wf    = explainer.expected_value
base_val = float(ev_wf[1]) if hasattr(ev_wf, '__len__') else float(ev_wf)

feat_labels = [f"{NOMBRES_VARIABLES[f]} = {pacientes_nuevos.iloc[paciente_idx][f]}"
               for f in FEATURES]

orden_wf   = np.argsort(np.abs(shap_carlos))
vals_ord   = shap_carlos[orden_wf]
labels_ord = [feat_labels[i] for i in orden_wf]
colores_wf = ["#F0997B" if v > 0 else "#5DCAA5" for v in vals_ord]

acumulado = base_val
starts    = []
for v in vals_ord:
    starts.append(acumulado)
    acumulado += v

for j, (start, val, label, color) in enumerate(
        zip(starts, vals_ord, labels_ord, colores_wf)):
    ax2.barh(j, val, left=start, color=color,
             edgecolor="white", linewidth=0.5, height=0.6)
    signo = "+" if val > 0 else ""
    ax2.text(start + val + (0.005 if val >= 0 else -0.005), j,
             f"{signo}{val*100:.1f}%", va="center",
             ha="left" if val >= 0 else "right", fontsize=8.5)

ax2.axvline(base_val, color="#888", linestyle="--", linewidth=1, alpha=0.7,
            label=f"Base: {base_val*100:.1f}%")
ax2.axvline(probabilidades[0], color="#D85A30", linestyle="-", linewidth=1.5,
            label=f"Predicción final: {probabilidades[0]*100:.1f}%")

ax2.set_yticks(range(len(labels_ord)))
ax2.set_yticklabels(labels_ord, fontsize=9)
ax2.set_xlabel("Probabilidad de riesgo cardiovascular", fontsize=10)
ax2.set_title(f"Explicación SHAP — {nombres[0]}\n¿Por qué el modelo dice RIESGO ALTO?",
              fontsize=12, fontweight="bold", pad=12)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=9, framealpha=0.6)

leyenda2 = [
    mpatches.Patch(color="#F0997B", label="Aumenta el riesgo"),
    mpatches.Patch(color="#5DCAA5", label="Reduce el riesgo"),
]
ax2.legend(handles=leyenda2 + ax2.get_legend_handles_labels()[0],
           fontsize=9, framealpha=0.6)
plt.tight_layout()
plt.savefig("grafica_2_waterfall_carlos.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Gráfica 3: Comparación de los 3 pacientes ──
fig3, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig3.patch.set_facecolor("#f8f8f8")
fig3.suptitle("Comparación SHAP entre los 3 pacientes\nCada barra = cuánto empujó esa variable hacia la decisión",
              fontsize=12, fontweight="bold")

for idx, (ax_i, nombre_i) in enumerate(zip(axes, nombres)):
    ax_i.set_facecolor("#f8f8f8")
    shap_i  = sv[idx]
    orden_i = np.argsort(np.abs(shap_i))
    vals_i  = shap_i[orden_i]
    labs_i  = [NOMBRES_VARIABLES[FEATURES[k]] for k in orden_i]
    cols_i  = ["#F0997B" if v > 0 else "#5DCAA5" for v in vals_i]

    ax_i.barh(labs_i, vals_i, color=cols_i,
              edgecolor="white", linewidth=0.5, height=0.6)
    ax_i.axvline(0, color="#888", linewidth=0.8)
    ax_i.set_title(f"{nombre_i}\n{'RIESGO ALTO' if predicciones[idx]==1 else ' RIESGO BAJO'} "
                   f"({probabilidades[idx]*100:.0f}%)",
                   fontsize=9, pad=8)
    ax_i.spines[["top", "right"]].set_visible(False)
    ax_i.set_xlabel("Valor SHAP", fontsize=8)

plt.tight_layout()
plt.savefig("grafica_3_comparacion.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# PASO 7: Reflexión final (las limitaciones)
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  CONCLUSIONES")
print("=" * 55)
print()
print("  Lo que SHAP SÍ hace:")
print("     - Explica cuánto influyó cada variable")
print("     - Funciona con cualquier modelo de ML")
print("     - Permite detectar sesgos o errores")
print()
print("  Lo que SHAP NO hace:")
print("     - No garantiza que el modelo esté bien")
print("     - Explicar ≠ tener razón")
print("     - Se puede construir un modelo injusto")
print("       con explicaciones que parecen correctas")
print()
print("  Archivos generados:")
print("     - grafica_1_importancia_global.png")
print("     - grafica_2_waterfall_carlos.png")
print("     - grafica_3_comparacion.png")
print()
print("=" * 55)
