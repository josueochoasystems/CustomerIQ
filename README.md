# CustomerIQ — Sistema Interactivo de Análisis de Clientes con ML

Aplicación web interactiva construida con **Streamlit** para análisis completo de datasets de clientes: exploración, limpieza, segmentación (clustering) y clasificación supervisada, con interpretación orientada a gerencia.

---

## ✦ Características principales

| Feature | Detalle |
|--------|---------|
| 🔌 **Genérico** | Funciona con cualquier dataset CSV / XLSX / JSON / TSV |
| 🎯 **Segmentación** | K-Means + Clustering Jerárquico evaluados con índice Silhouette |
| 🤖 **Clasificación** | Árbol de Decisión + Random Forest con Accuracy, F1, AUC y curva ROC |
| 🛡️ **Anti data leakage** | Escalado y encoding ajustados sólo sobre entrenamiento |
| 📋 **Historial** | Experimentos exportables a CSV |
| 🔮 **Predicción** | Individual y por lote + exportación de modelo `.pkl` |
| 📑 **Reporte ejecutivo** | Descargable en HTML |

---

## ⚙️ Instalación

```bash
# 1. Clonar repositorio
git clone <url-del-repo>
cd CustomerIQ

# 2. Crear entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## ▶ Ejecución

```bash
streamlit run app.py
```

Abrir en el navegador: `http://localhost:8501`

---

## 🗺 Flujo recomendado

| Paso | Página | Descripción |
|:----:|--------|-------------|
| `01` | 📁 Cargar Datos | Subir CSV, Excel, JSON o TSV |
| `02` | 🔍 Exploración | EDA automático: distribuciones, correlaciones, nulos |
| `03` | 🧹 Limpieza | Imputación, outliers y duplicados |
| `04` | ✂️ Partición | Train/Val/Test + encoding + escalado + baseline |
| `05` | 🎯 Segmentación | K-Means y clustering jerárquico |
| `06` | 🤖 Clasificación | Árbol de Decisión y Random Forest |
| `07` | 📊 Comparación | ROC, matrices de confusión, interpretación gerencial |
| `08` | 📑 Reporte | Resumen ejecutivo y descargas |

---

## 🗂 Estructura del proyecto

```
CustomerIQ/
├── app.py                  # Página principal y session_state global
├── requirements.txt
├── README.md
├── .gitignore
├── pages/                  # Páginas Streamlit multipágina
│   ├── 1_📁_Cargar_Datos.py
│   ├── 2_🔍_Exploracion.py
│   ├── 3_🧹_Limpieza.py
│   ├── 4_✂️_Particion.py
│   ├── 5_🎯_Segmentacion.py
│   ├── 6_🤖_Clasificacion.py
│   ├── 7_📊_Comparacion.py
│   └── 8_📑_Reporte.py
├── src/                    # Lógica de negocio (independiente de Streamlit)
│   ├── data_loader.py
│   ├── data_profiler.py
│   ├── cleaner.py
│   ├── preprocessor.py
│   ├── clustering.py
│   ├── classification.py
│   ├── evaluator.py
│   └── visualizer.py
├── models/                 # Modelos entrenados (.pkl) — generado en runtime
├── data/
│   ├── uploads/            # Archivos subidos temporalmente
│   └── processed/          # Datasets procesados
└── reports/                # Reportes HTML generados
```

---

## 🧰 Stack tecnológico

| Librería | Versión | Uso |
|----------|---------|-----|
| **Python** | 3.10+ | Lenguaje base |
| **Streamlit** | 1.35 | Framework UI multipágina |
| **scikit-learn** | 1.5 | ML: clustering, clasificación, preprocesado |
| **pandas / numpy** | — | Manipulación de datos |
| **plotly** | — | Visualizaciones interactivas |
| **scipy** | — | Dendrogramas |
| **joblib** | — | Persistencia de modelos |
| **openpyxl** | — | Soporte Excel |

---

## 💡 Dataset de ejemplo

Puedes usar el dataset **Mall Customers** (disponible en Kaggle) para probar la aplicación. Cualquier dataset tabular con al menos una columna numérica y una columna objetivo binaria o multiclase funciona.
