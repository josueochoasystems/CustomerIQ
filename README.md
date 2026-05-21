CustomerIQ — Sistema Interactivo de Análisis de Clientes con ML
Aplicación web interactiva construida con Streamlit para análisis completo de datasets de clientes: exploración, limpieza, segmentación (clustering) y clasificación supervisada, con interpretación orientada a gerencia.

✦ Características principales
FeatureDetalle🔌 GenéricoFunciona con cualquier dataset CSV / XLSX / JSON / TSV🎯 SegmentaciónK-Means + Clustering Jerárquico evaluados con índice Silhouette🤖 ClasificaciónÁrbol de Decisión + Random Forest con Accuracy, F1, AUC y curva ROC🛡️ Anti data leakageEscalado y encoding ajustados sólo sobre entrenamiento📋 HistorialExperimentos exportables a CSV🔮 PredicciónIndividual y por lote + exportación de modelo .pkl📑 Reporte ejecutivoDescargable en HTML

⚙️ Instalación
bash# 1. Clonar repositorio
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
▶ Ejecución
bashstreamlit run app.py
Abrir en el navegador: http://localhost:8501

🗺 Flujo recomendado
PasoPáginaDescripción01📁 Cargar DatosSubir CSV, Excel, JSON o TSV02🔍 ExploraciónEDA automático: distribuciones, correlaciones, nulos03🧹 LimpiezaImputación, outliers y duplicados04✂️ ParticiónTrain/Val/Test + encoding + escalado + baseline05🎯 SegmentaciónK-Means y clustering jerárquico06🤖 ClasificaciónÁrbol de Decisión y Random Forest07📊 ComparaciónROC, matrices de confusión, interpretación gerencial08📑 ReporteResumen ejecutivo y descargas

🗂 Estructura del proyecto
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

🧰 Stack tecnológico
LibreríaVersiónUsoPython3.10+Lenguaje baseStreamlit1.35Framework UI multipáginascikit-learn1.5ML: clustering, clasificación, preprocesadopandas / numpy—Manipulación de datosplotly—Visualizaciones interactivasscipy—Dendrogramasjoblib—Persistencia de modelosopenpyxl—Soporte Excel

💡 Dataset de ejemplo
Puedes usar el dataset Mall Customers (disponible en Kaggle) para probar la aplicación. Cualquier dataset tabular con al menos una columna numérica y una columna objetivo binaria o multiclase funciona.