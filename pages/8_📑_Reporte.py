"""Página 8 — Reporte ejecutivo: HTML, PDF y exportación de modelos y datasets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import os
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF

from src.evaluator import evaluate_classifier, experiments_to_dataframe

# ---------------------------------------------------------------------------
# Utilidades de texto para PDF
# ---------------------------------------------------------------------------
_UNICODE_MAP = {
    "—": "-", "–": "-",       # em dash, en dash
    "’": "'", "‘": "'",       # comillas tipograficas
    "“": '"', "”": '"',
    "•": "-",                      # bullet
    "°": " grados",
    "±": "+/-",
    "×": "x",
    "÷": "/",
    "≥": ">=", "≤": "<=",
    "é": "e", "ó": "o",       # por si acaso
    "ú": "u", "í": "i",
    "á": "a",
    "ñ": "n",
    "É": "E", "Ó": "O",
    "Ú": "U", "Í": "I",
    "Á": "A", "Ñ": "N",
    "ü": "u", "Ü": "U",
    "à": "a", "è": "e",
    "ì": "i", "ò": "o",
    "ù": "u",
}

# Rango de emojis y símbolos fuera de Latin-1 que no tienen sustituto textual
_EMOJI_RANGES = [
    (0x1F300, 0x1FAFF),   # Misc symbols, emojis
    (0x2600,  0x26FF),    # Misc symbols
    (0x2700,  0x27BF),    # Dingbats
    (0xFE00,  0xFE0F),    # Variation selectors
    (0x200D,  0x200D),    # Zero-width joiner
]


def _sanitize(text: str) -> str:
    """Convierte texto Unicode a cadena segura para fuentes PDF Latin-1.

    Sustituye caracteres problemáticos por equivalentes ASCII y elimina
    emojis o símbolos fuera del rango soportado.

    Args:
        text: Cadena de entrada con cualquier carácter Unicode.

    Returns:
        Cadena limpia compatible con fuentes PDF estándar.
    """
    if not isinstance(text, str):
        text = str(text)
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    result = []
    for ch in text:
        cp = ord(ch)
        in_emoji = any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES)
        if in_emoji:
            continue
        try:
            ch.encode("latin-1")
            result.append(ch)
        except UnicodeEncodeError:
            result.append("?")
    return "".join(result)


# Rutas de fuentes Arial en Windows
_FONT_DIR    = r"C:\Windows\Fonts"
_ARIAL_R     = os.path.join(_FONT_DIR, "arial.ttf")
_ARIAL_B     = os.path.join(_FONT_DIR, "arialbd.ttf")
_ARIAL_I     = os.path.join(_FONT_DIR, "ariali.ttf")
_ARIAL_BI    = os.path.join(_FONT_DIR, "arialbi.ttf")
_USE_ARIAL   = os.path.exists(_ARIAL_R)
_USE_ARIAL_BI = _USE_ARIAL and os.path.exists(_ARIAL_BI)

st.set_page_config(page_title="Reporte — CustomerIQ", layout="wide")

st.title("📑 Reporte Ejecutivo")

if st.session_state.get("df_raw") is None:
    st.warning("No hay datos cargados. Comienza en la página **1 — Cargar Datos**.")
    st.stop()

models        = st.session_state.get("models_trained", {})
has_partition = st.session_state.get("X_train") is not None
has_models    = bool(models)
df_raw        = st.session_state.df_raw
df_clean      = st.session_state.df_clean
file_name     = st.session_state.get("file_name", "—")
target_col    = st.session_state.get("target_col", "—")


# ---------------------------------------------------------------------------
# Resumen ejecutivo en pantalla
# ---------------------------------------------------------------------------
st.markdown("## Resumen Ejecutivo")
st.markdown(
    f"**Fecha del análisis:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}  \n"
    f"**Dataset analizado:** `{file_name}`"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros originales", f"{df_raw.shape[0]:,}")
col2.metric("Variables originales", df_raw.shape[1])
if df_clean is not None:
    col3.metric("Registros tras limpieza", f"{df_clean.shape[0]:,}")
    col4.metric("Filas eliminadas", f"{df_raw.shape[0] - df_clean.shape[0]:,}")
else:
    col3.metric("Limpieza", "No aplicada")
    col4.metric("—", "—")

if has_partition:
    st.markdown("### Partición")
    p1, p2, p3 = st.columns(3)
    p1.metric("Train", f"{len(st.session_state.X_train):,}")
    p2.metric("Validación", f"{len(st.session_state.X_val):,}")
    p3.metric("Test", f"{len(st.session_state.X_test):,}")
    st.markdown(f"**Variable objetivo:** `{target_col}`")
    st.markdown(f"**Features:** `{len(st.session_state.get('feature_cols', []))}` columnas")

if has_models and has_partition:
    st.markdown("### Rendimiento de modelos (test)")
    rows = []
    for name, model in models.items():
        mets = evaluate_classifier(model, st.session_state.X_test, st.session_state.y_test)
        rows.append({"Modelo": name, **mets})
    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    metric_cols = [c for c in ["f1", "accuracy", "auc"] if c in results_df.columns]
    if metric_cols:
        best_idx  = results_df[metric_cols[0]].idxmax()
        best_name = results_df.loc[best_idx, "Modelo"]
        best_f1   = results_df.loc[best_idx, metric_cols[0]]
        st.success(f"**Mejor modelo:** {best_name} ({metric_cols[0].upper()} = {best_f1:.4f})")

if st.session_state.get("clustering_results"):
    st.markdown("### Segmentación")
    for algo, res in st.session_state.clustering_results.items():
        algo_name = "K-Means" if algo == "kmeans" else "Clustering Jerárquico"
        n_cl = res.get("k") or res.get("n_clusters")
        sil  = res.get("silhouette", 0)
        st.markdown(f"- **{algo_name}**: {n_cl} clústeres — Silhouette: {sil:.4f}")


# ---------------------------------------------------------------------------
# Helpers de generación de informes
# ---------------------------------------------------------------------------

def _metrics_chart_bytes() -> bytes | None:
    """Genera un gráfico de barras de métricas y lo retorna como PNG en bytes."""
    if not (has_models and has_partition):
        return None
    rows = []
    for name, model in models.items():
        mets = evaluate_classifier(model, st.session_state.X_test, st.session_state.y_test)
        rows.append({"Modelo": name, **mets})
    df_m = pd.DataFrame(rows)
    metric_cols = [c for c in ["accuracy", "f1", "precision", "recall"] if c in df_m.columns]
    if not metric_cols:
        return None

    x = np.arange(len(df_m))
    width = 0.8 / len(metric_cols)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, metric in enumerate(metric_cols):
        ax.bar(x + i * width, df_m[metric], width, label=metric.capitalize(), color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(df_m["Modelo"], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas por modelo")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_html_report() -> str:
    """Construye el informe en HTML."""
    now      = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    n_raw    = df_raw.shape[0]
    n_clean  = df_clean.shape[0] if df_clean is not None else "N/A"

    model_rows = ""
    if has_models and has_partition:
        for name, model in models.items():
            mets    = evaluate_classifier(model, st.session_state.X_test, st.session_state.y_test)
            auc_str = f"{mets['auc']:.4f}" if mets["auc"] is not None else "N/A"
            model_rows += (
                f"<tr><td>{name}</td><td>{mets['accuracy']:.4f}</td>"
                f"<td>{mets['f1']:.4f}</td><td>{mets['precision']:.4f}</td>"
                f"<td>{mets['recall']:.4f}</td><td>{auc_str}</td></tr>"
            )

    clustering_rows = ""
    for algo, res in st.session_state.get("clustering_results", {}).items():
        algo_name = "K-Means" if algo == "kmeans" else "Clustering Jerárquico"
        n_cl = res.get("k") or res.get("n_clusters")
        sil  = res.get("silhouette", 0)
        clustering_rows += f"<tr><td>{algo_name}</td><td>{n_cl}</td><td>{sil:.4f}</td></tr>"

    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Reporte CustomerIQ — {file_name}</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:900px;margin:auto;padding:30px;color:#333}}
  h1{{color:#1a1a2e}} h2{{color:#16213e;border-bottom:2px solid #0f3460;padding-bottom:5px}}
  table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
  th{{background:#0f3460;color:white;padding:10px;text-align:left}}
  td{{border:1px solid #ddd;padding:8px}} tr:nth-child(even){{background:#f9f9f9}}
  .kpi{{display:inline-block;background:#e8f4f8;border-radius:8px;padding:15px 25px;margin:5px;text-align:center}}
  .kpi-val{{font-size:2em;font-weight:bold;color:#0f3460}}
  footer{{margin-top:40px;font-size:.8em;color:#999;text-align:center}}
</style></head><body>
<h1>🧠 CustomerIQ — Reporte de Análisis de Clientes</h1>
<p><strong>Fecha:</strong> {now} &nbsp;|&nbsp; <strong>Dataset:</strong> {file_name}</p>
<h2>1. Resumen del Dataset</h2>
<div>
  <div class="kpi"><div class="kpi-val">{n_raw:,}</div>Registros originales</div>
  <div class="kpi"><div class="kpi-val">{n_clean}</div>Registros tras limpieza</div>
  <div class="kpi"><div class="kpi-val">{target_col}</div>Variable objetivo</div>
</div>
<h2>2. Rendimiento de Clasificación (Test)</h2>
{"<table><tr><th>Modelo</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>AUC</th></tr>" + model_rows + "</table>" if model_rows else "<p>No se entrenaron modelos.</p>"}
<h2>3. Segmentación</h2>
{"<table><tr><th>Algoritmo</th><th>Clústeres</th><th>Silhouette</th></tr>" + clustering_rows + "</table>" if clustering_rows else "<p>No se ejecutó segmentación.</p>"}
<footer>Generado automáticamente por CustomerIQ</footer>
</body></html>"""


class _PDF(FPDF):
    """PDF con cabecera corporativa, pie de página y soporte Unicode via Arial TTF."""

    def __init__(self, title: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._report_title = _sanitize(title)
        self.alias_nb_pages()
        self._load_fonts()

    def _load_fonts(self):
        """Carga Arial TTF (Unicode) si está disponible en el sistema."""
        if _USE_ARIAL:
            self.add_font("Arial", style="",   fname=_ARIAL_R)
            self.add_font("Arial", style="B",  fname=_ARIAL_B)
            self.add_font("Arial", style="I",  fname=_ARIAL_I)
            if os.path.exists(_ARIAL_BI):
                self.add_font("Arial", style="BI", fname=_ARIAL_BI)
            self._fn = "Arial"
        else:
            # Helvetica soporta Latin-1; _sanitize() ya limpia el texto
            self._fn = "Helvetica"

    def _sf(self, style: str = "", size: int = 10):
        """Atajo para set_font con la fuente activa."""
        # Fall back to B if BI variant wasn't registered
        if self._fn == "Arial" and style == "BI" and not _USE_ARIAL_BI:
            style = "B"
        self.set_font(self._fn, style=style, size=size)

    def _avail_w(self) -> float:
        """Ancho disponible desde la posición x actual hasta el margen derecho."""
        return self.w - self.r_margin - self.x

    def _cell(self, w, h, txt, **kw):
        """cell() con sanitización automática de texto."""
        super().cell(w, h, _sanitize(str(txt)), **kw)

    def _mcell(self, w, h, txt, **kw):
        """multi_cell() con sanitización automática de texto."""
        super().multi_cell(w, h, _sanitize(str(txt)), **kw)

    def header(self):
        self.set_fill_color(15, 52, 96)
        self.set_text_color(255, 255, 255)
        self._sf("B", 11)
        self._cell(0, 10, self._report_title, fill=True,
                   new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-14)
        self._sf("I", 8)
        self.set_text_color(120, 120, 120)
        self._cell(0, 10, f"Pagina {self.page_no()}/{{nb}} - CustomerIQ", align="C")

    def section_title(self, text: str):
        self.set_x(self.l_margin)
        self._sf("B", 12)
        self.set_fill_color(220, 230, 245)
        self.set_text_color(15, 52, 96)
        self._cell(self._avail_w(), 8, text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, key: str, value: str):
        self.set_x(self.l_margin)   # Siempre empezar en margen izquierdo
        self._sf("B", 10)
        self._cell(70, 7, key + ":", new_x="RIGHT", new_y="TOP")
        self._sf("", 10)
        self._mcell(self._avail_w(), 7, str(value))

    def table(self, headers: list[str], rows: list[list], col_widths: list[float] | None = None):
        """Tabla con cabecera azul y filas alternadas."""
        if col_widths is None:
            avail = self.w - self.l_margin - self.r_margin
            col_widths = [avail / len(headers)] * len(headers)

        # Cabecera
        self.set_fill_color(15, 52, 96)
        self.set_text_color(255, 255, 255)
        self._sf("B", 9)
        for i, h in enumerate(headers):
            self._cell(col_widths[i], 7, str(h), border=1, fill=True, align="C")
        self.ln()

        # Filas
        self.set_text_color(0, 0, 0)
        self._sf("", 9)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell_val in enumerate(row):
                self._cell(col_widths[i], 6, str(cell_val), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


def _generate_pdf_report() -> bytes:
    """Genera el informe completo en PDF y retorna los bytes.

    Returns:
        Bytes del archivo PDF listo para descargar.
    """
    now    = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    n_raw  = df_raw.shape[0]
    n_clean_val = df_clean.shape[0] if df_clean is not None else "N/A"

    pdf = _PDF(title="CustomerIQ - Reporte de Analisis de Clientes con ML")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- PORTADA ----
    pdf.ln(10)
    pdf._sf("B", 20)
    pdf.set_text_color(15, 52, 96)
    pdf._cell(0, 12, "CustomerIQ", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf._sf("", 13)
    pdf.set_text_color(80, 80, 80)
    pdf._cell(0, 8, "Sistema Interactivo de Analisis de Clientes con ML",
              align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(15, 52, 96)
    pdf.set_line_width(0.8)
    pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
    pdf.ln(8)

    pdf._sf("", 10)
    pdf.set_text_color(0, 0, 0)
    pdf._cell(0, 6, f"Fecha de generacion: {now}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf._cell(0, 6, f"Dataset analizado: {_sanitize(file_name)}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # ---- SECCIÓN 1: Dataset ----
    pdf.add_page()
    pdf.section_title("1. Resumen del Dataset")
    pdf.kv_row("Archivo",                    _sanitize(file_name))
    pdf.kv_row("Registros originales",       f"{n_raw:,}")
    pdf.kv_row("Variables originales",       str(df_raw.shape[1]))
    pdf.kv_row("Registros tras limpieza",    str(n_clean_val))
    pdf.kv_row("Variable objetivo",          _sanitize(str(target_col)))
    if has_partition:
        pdf.kv_row("Features usadas", str(len(st.session_state.get("feature_cols", []))))
        pdf.kv_row("Tamano Train",    f"{len(st.session_state.X_train):,}")
        pdf.kv_row("Tamano Val",      f"{len(st.session_state.X_val):,}")
        pdf.kv_row("Tamano Test",     f"{len(st.session_state.X_test):,}")
    pdf.ln(4)

    # Calidad de datos
    pdf.section_title("2. Calidad de Datos")
    n_nulls = int(df_raw.isnull().sum().sum())
    n_dups  = int(df_raw.duplicated().sum())
    pdf.kv_row("Valores nulos totales (raw)", str(n_nulls))
    pdf.kv_row("Filas duplicadas (raw)",      str(n_dups))
    if df_clean is not None:
        n_nulls_c = int(df_clean.isnull().sum().sum())
        pdf.kv_row("Valores nulos tras limpieza", str(n_nulls_c))
    pdf.ln(4)

    # ---- SECCIÓN 2: Clasificación ----
    if has_models and has_partition:
        pdf.section_title("3. Resultados de Clasificación (conjunto de test)")
        headers = ["Modelo", "Accuracy", "F1", "Precision", "Recall", "AUC"]
        col_w   = [45, 22, 22, 22, 22, 22]
        rows_data = []
        best_f1_val, best_name_val = -1.0, ""
        for name, model in models.items():
            mets = evaluate_classifier(model, st.session_state.X_test, st.session_state.y_test)
            auc_str = f"{mets['auc']:.4f}" if mets["auc"] is not None else "N/A"
            rows_data.append([
                name,
                f"{mets['accuracy']:.4f}",
                f"{mets['f1']:.4f}",
                f"{mets['precision']:.4f}",
                f"{mets['recall']:.4f}",
                auc_str,
            ])
            if mets["f1"] > best_f1_val:
                best_f1_val  = mets["f1"]
                best_name_val = name

        pdf.table(headers, rows_data, col_widths=col_w)

        pdf._sf("BI", 10)
        pdf.set_text_color(15, 52, 96)
        pdf._cell(0, 7,
                  f"Modelo recomendado (mayor F1): {_sanitize(best_name_val)} - F1 = {best_f1_val:.4f}",
                  new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

        # Gráfico de métricas embebido
        chart_bytes = _metrics_chart_bytes()
        if chart_bytes:
            pdf.section_title("4. Gráfico comparativo de métricas")
            chart_buf = io.BytesIO(chart_bytes)
            pdf.image(chart_buf, x=15, w=pdf.w - 30)
            pdf.set_x(pdf.l_margin)  # reset x tras imagen
            pdf.ln(4)

    # ---- SECCIÓN 3: Segmentación ----
    clustering_results = st.session_state.get("clustering_results", {})
    section_num = 5 if (has_models and has_partition) else 3
    if clustering_results:
        pdf.section_title(f"{section_num}. Resultados de Segmentación (Clustering)")
        headers_cl = ["Algoritmo", "Clústeres", "Silhouette Score"]
        rows_cl    = []
        for algo, res in clustering_results.items():
            algo_name = "K-Means" if algo == "kmeans" else "Clustering Jerárquico"
            n_cl = res.get("k") or res.get("n_clusters")
            sil  = res.get("silhouette", 0)
            rows_cl.append([algo_name, str(n_cl), f"{sil:.4f}"])
        pdf.table(headers_cl, rows_cl, col_widths=[70, 40, 47])

        pdf._sf("I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.set_x(pdf.l_margin)
        pdf._mcell(pdf.w - pdf.l_margin - pdf.r_margin, 6,
            "El Silhouette Score varia entre -1 y 1. "
            "Valores cercanos a 1 indican clusteres bien definidos y separados."
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # ---- SECCIÓN 4: Historial de experimentos ----
    exp_df = experiments_to_dataframe()
    if exp_df is not None and not exp_df.empty:
        sec = section_num + (1 if clustering_results else 0)
        pdf.add_page()
        pdf.section_title(f"{sec}. Historial de Experimentos")
        exp_cols    = ["model", "split", "accuracy", "f1", "precision", "recall", "auc"]
        exp_cols_ok = [c for c in exp_cols if c in exp_df.columns]
        headers_exp = [c.capitalize() for c in exp_cols_ok]
        n_cols      = len(headers_exp)
        avail       = pdf.w - pdf.l_margin - pdf.r_margin
        col_w_exp   = [avail / n_cols] * n_cols
        rows_exp    = []
        for _, row in exp_df.iterrows():
            r = []
            for c in exp_cols_ok:
                val = row[c]
                r.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            rows_exp.append(r)
        pdf.table(headers_exp, rows_exp, col_widths=col_w_exp)

    # ---- SECCIÓN FINAL: Interpretación gerencial ----
    pdf.add_page()
    pdf.section_title("Interpretacion para la Gerencia")
    if has_models and has_partition and best_name_val:
        model_best = models.get(best_name_val)
        if model_best:
            mets_b = evaluate_classifier(model_best, st.session_state.X_test, st.session_state.y_test)
            acc  = mets_b["accuracy"]
            prec = mets_b["precision"]
            rec  = mets_b["recall"]
            f1   = mets_b["f1"]
            nombre = _sanitize(best_name_val)

            pdf._sf("", 10)
            lines = [
                f"El modelo seleccionado es '{nombre}', evaluado sobre el conjunto de prueba.",
                "",
                f"  - De cada 100 clientes analizados, el modelo clasifica correctamente a "
                f"{int(acc*100)} (Accuracy = {acc:.1%}).",
                f"  - De los clientes identificados como positivos, {int(prec*100)} de cada 100 "
                f"realmente lo son (Precision = {prec:.1%}).",
                f"  - De los clientes que realmente son positivos, el modelo detecta a "
                f"{int(rec*100)} de cada 100 (Recall = {rec:.1%}).",
                f"  - El balance F1 entre precision y deteccion es {f1:.2f} sobre 1.0.",
            ]
            baseline_obj = models.get("Baseline")
            if baseline_obj:
                bl_acc  = evaluate_classifier(
                    baseline_obj, st.session_state.X_test, st.session_state.y_test
                )["accuracy"]
                mejora  = (acc - bl_acc) * 100
                calidad = "significativa" if mejora > 5 else "moderada"
                lines += [
                    "",
                    "Comparacion con el punto de partida (Baseline):",
                    f"  - Un sistema que predice siempre la clase mas comun acertaria el {bl_acc:.1%}.",
                    f"  - '{nombre}' mejora en {mejora:+.1f} puntos porcentuales ({calidad} mejora).",
                ]
            _full_w = pdf.w - pdf.l_margin - pdf.r_margin
            for line in lines:
                pdf.set_x(pdf.l_margin)
                pdf._mcell(_full_w, 6, line)
    else:
        pdf._sf("I", 10)
        pdf.set_x(pdf.l_margin)
        pdf._mcell(pdf.w - pdf.l_margin - pdf.r_margin, 7,
            "No hay modelos entrenados. Completa las paginas de Clasificacion para generar esta seccion.")

    pdf.ln(6)
    pdf._sf("I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf._cell(0, 6, "Generado automaticamente por CustomerIQ - Sistema de Analisis de Clientes con ML",
              align="C", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Sección de descargas
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Descargas")

col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

# PDF
with col_dl1:
    with st.spinner("Generando PDF..."):
        try:
            pdf_bytes = _generate_pdf_report()
            st.download_button(
                "📄 Descargar reporte PDF",
                data=pdf_bytes,
                file_name=f"reporte_customeriq_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error generando PDF: {e}")

# HTML
with col_dl2:
    html_bytes = _generate_html_report().encode("utf-8")
    st.download_button(
        "🌐 Descargar reporte HTML",
        data=html_bytes,
        file_name=f"reporte_customeriq_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        use_container_width=True,
    )

# Dataset procesado
with col_dl3:
    if df_clean is not None:
        st.download_button(
            "📊 Dataset procesado (CSV)",
            data=df_clean.to_csv(index=False).encode("utf-8"),
            file_name="dataset_procesado.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("📊 Dataset procesado", disabled=True, use_container_width=True,
                  help="Primero aplica limpieza y partición.")

# Modelo .pkl
with col_dl4:
    trained_non_baseline = {k: v for k, v in models.items() if k != "Baseline"}
    if trained_non_baseline:
        model_choice = st.selectbox("Modelo a exportar", list(trained_non_baseline.keys()))
        buf = io.BytesIO()
        joblib.dump(trained_non_baseline[model_choice], buf)
        buf.seek(0)
        st.download_button(
            "🤖 Descargar modelo (.pkl)",
            data=buf.read(),
            file_name=f"{model_choice.replace(' ', '_').lower()}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )
    else:
        st.button("🤖 Modelo (.pkl)", disabled=True, use_container_width=True,
                  help="Primero entrena un modelo en la página 6.")

# Historial de experimentos
st.markdown("### Historial de experimentos")
exp_df = experiments_to_dataframe()
if exp_df is not None:
    st.dataframe(exp_df, use_container_width=True)
    st.download_button(
        "📋 Descargar historial (CSV)",
        data=exp_df.to_csv(index=False).encode("utf-8"),
        file_name="historial_experimentos.csv",
        mime="text/csv",
    )
else:
    st.info("No hay experimentos registrados aún.")
