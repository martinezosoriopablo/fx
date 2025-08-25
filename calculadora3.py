
# streamlit_app.py (lee FX de assets + uploader; PDF + email; montos con miles - FIXED money_input)
# -------------------------------------------------------------
# Comparador de Pagos – Banco vs Cargo Produce by Vita
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import os, re
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from email.message import EmailMessage

# ========== Page / Branding ==========
st.set_page_config(
    page_title="Comparador de Pagos – Banco vs Cargo Produce by Vita",
    layout="wide"
)

# --- Assets ---
LOGO_VITA = "assets/logo_vita.jpg"
LOGO_CP   = "assets/logo_cargoproduce.jpg"
ASSET_FX  = "assets/tipos_cambio.xlsx"

# =============================
# Estado
# =============================
if "flujos" not in st.session_state:
    st.session_state.flujos = []
if "lineas" not in st.session_state:
    st.session_state.lineas = []
if "last_added" not in st.session_state:
    st.session_state.last_added = None
# -- Empresa
if "empresa_nombre" not in st.session_state:
    st.session_state.empresa_nombre = "Empresa Demo"
if "empresa_pais" not in st.session_state:
    st.session_state.empresa_pais = "Chile"
if "empresa_logo_bytes" not in st.session_state:
    st.session_state.empresa_logo_bytes = None

# -- Perfil Vita y Matriz
if "vita_profile" not in st.session_state:
    st.session_state.vita_profile = {"perfil": "Mediana", "fijo": 0.0, "pct": 0.0, "fx_pct": 0.30}
if "vita_matrix" not in st.session_state:
    st.session_state.vita_matrix = None

# -- FX rates (USD por 1 unidad de moneda)
if "fx_rates" not in st.session_state:
    st.session_state.fx_rates = {}

LATAM_DESTINOS = ["Chile", "Perú", "Colombia", "Argentina", "Brasil", "México", "Uruguay"]
ORIGENES = [
    ("Estados Unidos", "USD"),
    ("Europa", "EUR"),
    ("Reino Unido", "GBP"),
    ("China", "USD"),
    ("México", "USD"),
    ("Brasil", "USD"),
    ("Colombia", "USD"),
    ("Chile", "USD"),
    ("Perú", "USD"),
    ("Argentina", "USD"),
    ("Uruguay", "USD"),
]

# =============================
# Utils
# =============================
def parse_number(x, default=None):
    if x is None: return default
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace("$", "").replace(" ", "")
    s = re.sub(r"[^0-9,\.\-]", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) == 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except:
        return default

def money_input(label, value, key, help=None, digits: int = 0):
    """
    Campo de texto con separador de miles. Corrige el error:
    'StreamlitAPIException: st.session_state.<key> cannot be modified after the widget with key <key> is instantiated'
    Formateamos ANTES de instanciar el widget y nunca escribimos después.
    """
    # 1) Definir/ajustar el valor en session_state *antes* de crear el widget
    default_formatted = f"{float(value):,.{digits}f}"
    if key not in st.session_state:
        st.session_state[key] = default_formatted
    else:
        prev = st.session_state[key]
        parsed = parse_number(prev, default=float(value))
        st.session_state[key] = f"{parsed:,.{digits}f}"

    # 2) Crear el widget (no pasamos 'value' para respetar session_state)
    txt = st.text_input(label, key=key, help=help)

    # 3) Parsear y devolver número
    val = parse_number(txt, default=float(value))
    return float(val or 0.0)

def load_fx_rates_from_assets(path=ASSET_FX):
    """
    Lee assets/tipos_cambio.xlsx con columnas:
    - Par (ej: EUR/USD, USD/CLP)
    - Valor (numérico)
    Reglas:
    - XXX/USD => map[XXX] = Valor  (USD por 1 XXX)
    - USD/YYY => map[YYY] = 1/Valor (USD por 1 YYY)
    """
    fx_map = {}
    if not os.path.exists(path):
        return fx_map
    try:
        df = pd.read_excel(path, sheet_name=0)
        df.columns = [str(c).strip().title() for c in df.columns]
        if not {"Par", "Valor"}.issubset(df.columns):
            return fx_map
        for _, r in df.iterrows():
            pair = str(r["Par"]).strip().upper()
            val = parse_number(r["Valor"], default=None)
            if not pair or val is None or val <= 0:
                continue
            if "/" not in pair:
                continue
            base, quote = [p.strip() for p in pair.split("/", 1)]
            if quote == "USD":
                fx_map[base] = float(val)
            elif base == "USD":
                fx_map[quote] = 1.0 / float(val)
    except Exception:
        return {}
    return fx_map

def merge_fx_maps(primary, fallback):
    out = dict(fallback)
    out.update({k:v for k,v in (primary or {}).items() if v})
    return out

# ---------- Utilidades de ruteo ----------
def _price_col(df: pd.DataFrame) -> str:
    if "Tarifa al Cliente (USD)" in df.columns and not df["Tarifa al Cliente (USD)"].isna().all():
        return "Tarifa al Cliente (USD)"
    return "Costo CP-Vita (USD)"

def _speed_rank(val: str) -> int:
    val = (val or "").strip().lower()
    if "instant" in val: return 0
    if "minut"   in val: return 1
    if "hora"    in val: return 2
    if "día" in val or "dia" in val: return 3
    return 9

def pick_vita_route(origen: str, moneda_origen: str, destino: str, cuenta: str, mode: str = "Recomendado") -> dict | None:
    df = st.session_state.get("vita_matrix")
    if df is None or df.empty:
        return None
    cuenta_lower = (cuenta or "").lower()
    cand = df[(df["Origen"] == origen)]
    if "EUR" in moneda_origen: cand = cand[cand["Moneda"] == "EUR"]
    elif "GBP" in moneda_origen: cand = cand[cand["Moneda"] == "GBP"]
    else: cand = cand[cand["Moneda"] == "USD"]

    if origen == "Estados Unidos" and "cuenta en ee.uu." in cuenta_lower:
        cand = cand[(cand["Destino"] == "Estados Unidos") & (cand["Medio"].isin(["Wire", "ACH", "FedNow"]))]
    elif origen == "Europa" and "eur" in cuenta_lower:
        cand = cand[(cand["Destino"] == "Europa") & (cand["Medio"].isin(["SEPA", "SEPA Instant"]))]
    elif origen == "Reino Unido" and destino == "Reino Unido":
        cand = cand[(cand["Destino"] == "Reino Unido") & (cand["Medio"] == "Faster Payments")]
    elif origen == "China" and destino == "China":
        cand = cand[cand["Medio"] == "Stablecoin"]
    else:
        fuera = ("Fuera de EE.UU." if origen == "Estados Unidos"
                 else ("Fuera de Europa" if origen == "Europa"
                       else ("Fuera de Reino Unido" if origen == "Reino Unido" else f"Fuera de {origen}")))
        cand = cand[cand["Destino"] == fuera]
        if not cand.empty and (cand["Medio"] == "SWIFT").any():
            cand = cand[cand["Medio"] == "SWIFT"]

    if cand.empty:
        return None

    price_col = _price_col(cand)
    mode_norm = (mode or "Recomendado").strip().lower()
    if mode_norm.startswith("recom"):
        if cand["Recomendado"].any(): cand = cand[cand["Recomendado"] == True]
        cand = cand.sort_values([price_col], ascending=[True])
    elif "barat" in mode_norm:
        cand = cand.sort_values([price_col], ascending=[True])
    else:
        speeds = cand["Tiempo Estimado de Ejecución"].map(_speed_rank)
        cand = cand.assign(_speed=speeds).sort_values(["_speed", price_col], ascending=[True, True])

    row = cand.iloc[0]
    return {
        "fee": float(row[price_col]),
        "medio": str(row.get("Medio", "")),
        "tiempo": str(row.get("Tiempo Estimado de Ejecución", "")),
        "criterio": "Recomendado" if mode_norm.startswith("recom") else ("Más barato" if "barat" in mode_norm else "Más rápido"),
    }

# =============================
# Header
# =============================
# Header: título + logos
hc1, hc2 = st.columns([0.75, 0.05])
with hc1:
    st.title("Comparador de Pagos – Banco vs Cargo Produce by Vita")
with hc2:
    c_l1, c_l2 = st.columns(2)
    with c_l1:
        if os.path.exists(LOGO_VITA):
            st.image(LOGO_VITA, use_container_width=True)
    with c_l2:
        if os.path.exists(LOGO_CP):
            st.image(LOGO_CP, use_container_width=True)

st.caption("Agrega flujos uno por uno y comparte un resultado claro y simple con tus clientes.")

# =============================
# Tabs
# =============================
company_tab, inputs_tab, results_tab = st.tabs([
    "0) Configuración de la Empresa",
    "1) Inputs (líneas)",
    "2) Resultados"
])

# =============================
# 0) Empresa: y FX (assets + uploader)
# =============================
with company_tab:
    st.subheader("Configuración de la Empresa – Nombre, Logo, País y FX")

    st.session_state.empresa_nombre = st.text_input("Nombre de la empresa", value=st.session_state.empresa_nombre)
    up = st.file_uploader("Subir logo de la empresa (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up is not None:
        st.session_state.empresa_logo_bytes = up.read()
        st.success("Logo cargado.")

    st.session_state.empresa_pais = st.selectbox("País de origen de la empresa (LATAM)", LATAM_DESTINOS,
                                                 index=LATAM_DESTINOS.index(st.session_state.empresa_pais) if st.session_state.empresa_pais in LATAM_DESTINOS else 0)

    # Perfil
    perfiles = {
        "Pequeña": {"fijo": 0.0, "pct": 0.00, "fx_pct": 0.45},
        "Mediana": {"fijo": 0.0, "pct": 0.00, "fx_pct": 0.30},
        "Grande": {"fijo": 0.0, "pct": 0.00, "fx_pct": 0.18},
        "Multinacional": {"fijo": 0.0, "pct": 0.00, "fx_pct": 0.10},
    }
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        perfil_sel = st.selectbox("Perfil", list(perfiles.keys()),
                                  index=list(perfiles.keys()).index(st.session_state.vita_profile.get("perfil", "Mediana")))
    st.session_state.vita_profile.update(perfiles[perfil_sel])
    with c2: st.session_state.vita_profile["fijo"] = st.number_input("Fijo (USD)", min_value=0.0, value=float(st.session_state.vita_profile["fijo"]), step=0.5, format="%.2f")
    with c3: st.session_state.vita_profile["pct"]  = st.number_input("% sobre monto", min_value=0.0, value=float(st.session_state.vita_profile["pct"]), step=0.05, format="%.2f")
    with c4: st.session_state.vita_profile["fx_pct"] = st.number_input("FX % (si aplica)", min_value=0.0, value=float(st.session_state.vita_profile["fx_pct"]), step=0.05, format="%.2f")

    # Matriz costos (mantener)
    st.markdown("### Matriz de Costos por Canal (editable)")
    if st.session_state.vita_matrix is None:
        st.session_state.vita_matrix = pd.DataFrame([
            {"Origen": "Estados Unidos", "Moneda": "USD", "Destino": "Estados Unidos", "Medio": "Wire",           "Costo CP-Vita (USD)": 1.50, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True,  "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Estados Unidos", "Moneda": "USD", "Destino": "Estados Unidos", "Medio": "ACH",            "Costo CP-Vita (USD)": 0.50, "Tarifa al Cliente (USD)": np.nan, "Recomendado": False, "Tiempo Estimado de Ejecución": "1 a 3 días"},
            {"Origen": "Estados Unidos", "Moneda": "USD", "Destino": "Estados Unidos", "Medio": "FedNow",         "Costo CP-Vita (USD)": 0.50, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True,  "Tiempo Estimado de Ejecución": "Instantáneo"},
            {"Origen": "Estados Unidos", "Moneda": "USD", "Destino": "Fuera de EE.UU.", "Medio": "SWIFT",        "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True,  "Tiempo Estimado de Ejecución": "Horas"},

            {"Origen": "Europa", "Moneda": "EUR", "Destino": "Europa", "Medio": "SEPA",         "Costo CP-Vita (USD)": 1.00, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True,  "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Europa", "Moneda": "EUR", "Destino": "Europa", "Medio": "SEPA Instant", "Costo CP-Vita (USD)": 1.50, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True,  "Tiempo Estimado de Ejecución": "Instantáneo"},
            {"Origen": "Europa", "Moneda": "EUR", "Destino": "Fuera de Europa", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},

            {"Origen": "Reino Unido", "Moneda": "GBP", "Destino": "Reino Unido", "Medio": "Faster Payments", "Costo CP-Vita (USD)": 1.00, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Minutos"},
            {"Origen": "Reino Unido", "Moneda": "GBP", "Destino": "Fuera de Reino Unido", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},

            {"Origen": "China", "Moneda": "USD", "Destino": "China", "Medio": "Stablecoin", "Costo CP-Vita (USD)": 2.00, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Minutos"},

            {"Origen": "México", "Moneda": "USD", "Destino": "Fuera de México", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Brasil", "Moneda": "USD", "Destino": "Fuera de Brasil", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Colombia", "Moneda": "USD", "Destino": "Fuera de Colombia", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Chile", "Moneda": "USD", "Destino": "Fuera de Chile", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Perú", "Moneda": "USD", "Destino": "Fuera de Perú", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Argentina", "Moneda": "USD", "Destino": "Fuera de Argentina", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},
            {"Origen": "Uruguay", "Moneda": "USD", "Destino": "Fuera de Uruguay", "Medio": "SWIFT", "Costo CP-Vita (USD)": 15.0, "Tarifa al Cliente (USD)": np.nan, "Recomendado": True, "Tiempo Estimado de Ejecución": "Horas"},
        ])
    st.session_state.vita_matrix = st.data_editor(st.session_state.vita_matrix, use_container_width=True, num_rows="dynamic", hide_index=True)
    vm = st.session_state.vita_matrix.copy()
    if "Tarifa al Cliente (USD)" in vm.columns and "Costo CP-Vita (USD)" in vm.columns:
        mask_nan = vm["Tarifa al Cliente (USD)"].isna()
        vm.loc[mask_nan, "Tarifa al Cliente (USD)"] = vm.loc[mask_nan, "Costo CP-Vita (USD)"] + 3.0
        st.session_state.vita_matrix = vm

    # FX: cargar desde assets por defecto
    assets_fx = load_fx_rates_from_assets(ASSET_FX)
    # Uploader opcional que sobreescribe
    st.markdown("---")
    st.markdown("### 📥 Tipos de cambio")
    st.caption("Por defecto, se cargan desde **assets/tipos_cambio.xlsx** (columnas: Par, Valor). Puedes subir otro archivo para sobreescribir temporalmente.")
    fx_file = st.file_uploader("Subir Excel de tipos de cambio (.xlsx)", type=["xlsx"], key="uploader_fx")
    uploaded_fx = {}
    if fx_file is not None:
        try:
            dfu = pd.read_excel(fx_file, sheet_name=0)
            cols = [c.strip().title() for c in dfu.columns]
            dfu.columns = cols
            if {"Par","Valor"}.issubset(dfu.columns):
                for _, r in dfu.iterrows():
                    pair = str(r["Par"]).strip().upper()
                    val  = parse_number(r["Valor"], default=None)
                    if not pair or val is None or val <= 0: continue
                    if "/" not in pair: continue
                    base, quote = [p.strip() for p in pair.split("/",1)]
                    if quote == "USD":
                        uploaded_fx[base] = float(val)
                    elif base == "USD":
                        uploaded_fx[quote] = 1.0/float(val)
                st.success(f"FX cargado desde archivo ({len(uploaded_fx)} monedas).")
                st.dataframe(dfu, use_container_width=True)
            else:
                st.error("El Excel debe tener columnas: Par, Valor.")
        except Exception as e:
            st.error(f"No se pudo leer el Excel: {e}")

    # Plantilla descargable
    sample_fx = pd.DataFrame({
        "Par": ["EUR/USD","USD/CLP","USD/PEN","USD/BRL","USD/MXN","GBP/USD"],
        "Valor": [1.16, 969.90, 3.50, 5.47, 18.74, 1.34],
    })
    try:
        import io
        with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as writer:
            sample_fx.to_excel(writer, sheet_name="FX", index=False)
            bio = writer.book.filename  # BytesIO
        st.download_button("⬇️ Descargar plantilla FX (Excel)", data=bio.getvalue(), file_name="tipos_cambio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.download_button("⬇️ Descargar plantilla FX (CSV)", data=sample_fx.to_csv(index=False).encode("utf-8"), file_name="tipos_cambio.csv", mime="text/csv")

    st.session_state.fx_rates = merge_fx_maps(uploaded_fx, assets_fx)
    if st.session_state.fx_rates:
        st.caption("Monedas disponibles: " + ", ".join(sorted(st.session_state.fx_rates.keys())))
    else:
        st.warning("No se encontraron tipos de cambio. Se asume USD=1. Carga el Excel en assets/tipos_cambio.xlsx.", icon="⚠️")

st.markdown("---")

# =============================
# 1) Inputs (líneas) — monto origen + equivalente USD
# =============================
with inputs_tab:
    st.subheader("Paso 1 – Define tus flujos (cada fila es una línea)")

    st.session_state.setdefault("route_mode", "Recomendado")
    st.session_state.route_mode = st.radio("Criterio de ruta Vita", ["Recomendado","Más barato","Más rápido"],
                                           index=["Recomendado","Más barato","Más rápido"].index(st.session_state.get("route_mode","Recomendado")),
                                           horizontal=True)

    def default_line():
        return {
            "Monto origen": 5_000_000.0,
            "Tipo": "Recepción",
            "Origen": "Europa",
            "Moneda Origen": "EUR",
            "Destino": st.session_state.empresa_pais if st.session_state.empresa_pais in LATAM_DESTINOS else "Chile",
            "Cuenta": "Cuenta en EE.UU.",
            "Pagos": 25,
            "Banco fijo": 25.0,
            "Banco %": 0.0,
            "FX?": False,
            "FX % banco": 0.12,
        }

    ctop1, ctop2, ctop3 = st.columns([1,1,2])
    if ctop1.button("➕ Agregar línea"): st.session_state.lineas.append(default_line())
    if ctop2.button("➕➕ Agregar 3 líneas"): st.session_state.lineas.extend([default_line() for _ in range(3)])
    if ctop3.button("🗑️ Limpiar líneas"): st.session_state.lineas = []

    for idx, data in enumerate(st.session_state.lineas):
        st.markdown(f"**Línea {idx+1}**")
        c_monto, c_tipo, c_origen, c_dest, c_cuenta, c_pagos, c_bfijo, c_bpct, c_fxchk, c_fxval, c_equiv, c_rm = st.columns([1.1,0.8,1.0,1.0,1.2,0.9,1.1,1.0,0.7,1.0,1.3,0.5])
        with c_tipo:
            data["Tipo"] = st.selectbox("Tipo", ["Recepción","Envío"], index=["Recepción","Envío"].index(data.get("Tipo","Recepción")), key=f"tipo_{idx}")
        with c_origen:
            data["Origen"] = st.selectbox("Origen", [o[0] for o in ORIGENES], index=[o[0] for o in ORIGENES].index(data.get("Origen","Europa")), key=f"origen_{idx}")
            data["Moneda Origen"] = dict(ORIGENES)[data["Origen"]]
        with c_dest:
            data["Destino"] = st.selectbox("Destino", LATAM_DESTINOS, index=LATAM_DESTINOS.index(data.get("Destino", LATAM_DESTINOS[0])), key=f"dest_{idx}")
        with c_cuenta:
            data["Cuenta"] = st.selectbox("Se recibe en", ["Cuenta en EE.UU.","Cuenta local (USD)","Cuenta local (EUR)"],
                                          index=["Cuenta en EE.UU.","Cuenta local (USD)","Cuenta local (EUR)"].index(data.get("Cuenta","Cuenta en EE.UU.")),
                                          key=f"cuenta_{idx}")
        with c_pagos:
            default_pagos = int(max(1, (data.get("Monto origen", 0) or 0) // 50_000))
            data["Pagos"] = st.number_input("# Pagos", min_value=1, value=int(data.get("Pagos", default_pagos)), step=1, key=f"pagos_{idx}")
        with c_bfijo:
            data["Banco fijo"] = st.number_input("Banco fijo (all-in)", min_value=0.0, value=float(data.get("Banco fijo",85.0)), step=1.0, format="%.2f", key=f"bfijo_{idx}")
        with c_bpct:
            data["Banco %"] = st.number_input("Banco %", min_value=0.0, value=float(data.get("Banco %",0.5)), step=0.05, format="%.2f", key=f"bpct_{idx}")
        with c_fxchk:
            data["FX?"] = st.checkbox("FX?", value=bool(data.get("FX?", False)), key=f"fxchk_{idx}", help="Aplica costo de FX sobre el monto (USD).")
        with c_fxval:
            data["FX % banco"] = st.number_input("FX % banco", min_value=0.0, value=float(data.get("FX % banco",0.5)), step=0.05, format="%.2f", key=f"fxpct_{idx}")
        with c_monto:
            data["Monto origen"] = money_input(f"Monto ({data['Moneda Origen']})", value=float(data.get("Monto origen",0.0)), key=f"monto_origen_{idx}")
        with c_equiv:
            cur = data["Moneda Origen"]
            rate = st.session_state.fx_rates.get(cur, 1.0 if cur == "USD" else None)
            if rate is None:
                st.warning(f"Falta tipo de cambio para {cur} (cárgalo en assets o sube un Excel).", icon="⚠️")
                equiv_usd = float(data.get("Equivalente USD", 0.0))
            else:
                equiv_usd = float(data["Monto origen"]) * float(rate)
            data["Equivalente USD"] = equiv_usd
            st.metric("Equivalente en USD", f"${equiv_usd:,.0f}")
        with c_rm:
            if st.button("🗑️", key=f"rm_{idx}"):
                st.session_state.lineas.pop(idx)
                st.experimental_rerun()

    if st.button("✅ Calcular y guardar flujos"):
        st.session_state.flujos = []
        for data in st.session_state.lineas:
            _route = pick_vita_route(origen=data["Origen"], moneda_origen=data["Moneda Origen"], destino=data["Destino"], cuenta=data["Cuenta"], mode=st.session_state.get("route_mode","Recomendado"))
            vita_fee = float((_route or {}).get("fee", st.session_state.vita_profile["fijo"]))
            vita_medio = (_route or {}).get("medio", "")
            vita_tiempo = (_route or {}).get("tiempo", "")
            vita_criterio = (_route or {}).get("criterio", st.session_state.get("route_mode","Recomendado"))
            monto_usd = float(data.get("Equivalente USD", 0.0))
            st.session_state.flujos.append({
                "Empresa": st.session_state.empresa_nombre,
                "País de origen": st.session_state.empresa_pais,
                "Tipo": data["Tipo"],
                "Monto (USD)": monto_usd,
                "# Pagos": int(data["Pagos"]),
                "Origen": data["Origen"],
                "Moneda Origen": data["Moneda Origen"],
                "Destino": data["Destino"],
                "Cuenta recepción": data["Cuenta"],
                "Banco fijo": float(data["Banco fijo"]),
                "Banco %": float(data["Banco %"]),
                "Vita fijo": vita_fee,
                "Vita %": float(st.session_state.vita_profile["pct"]),
                "Ruta Vita": vita_medio,
                "Tiempo Ruta": vita_tiempo,
                "Criterio Ruta": vita_criterio,
                "FX aplica": bool(data["FX?"]),
                "FX Banco %": float(data["FX % banco"]),
                "FX Vita %": float(st.session_state.vita_profile["fx_pct"]),
                "Monto origen": float(data["Monto origen"]),
                "Equivalente USD": monto_usd,
            })
        st.session_state.last_added = len(st.session_state.flujos) - 1
        st.success("Flujos guardados. Ve a **Resultados** para mostrar el ahorro.")

# =============================
# PDF / Email helpers
# =============================
def _draw_kpi(c, x, y, title, value):
    c.setFont("Helvetica-Bold", 11); c.drawString(x, y, title)
    c.setFont("Helvetica", 12); c.drawString(x, y-14, value)

def build_pdf(company_name, country, logo_bytes, totals, df_cards) -> bytes:
    buffer = BytesIO(); width, height = A4
    c = canvas.Canvas(buffer, pagesize=A4); margin = 2*cm
    x = margin
    if os.path.exists(LOGO_VITA):
        try: c.drawImage(LOGO_VITA, x, height-2.2*cm, width=3.6*cm, preserveAspectRatio=True, mask='auto')
        except: pass
    if os.path.exists(LOGO_CP):
        try: c.drawImage(LOGO_CP, x+4.0*cm, height-2.2*cm, width=3.6*cm, preserveAspectRatio=True, mask='auto')
        except: pass
    if logo_bytes:
        try: c.drawImage(ImageReader(BytesIO(logo_bytes)), width-5.2*cm, height-2.2*cm, width=3.6*cm, preserveAspectRatio=True, mask='auto')
        except: pass
    c.setFont("Helvetica-Bold", 16); c.drawString(margin, height-3.0*cm, "Comparador de Pagos – Banco vs CP by Vita")
    c.setFont("Helvetica", 12); c.drawString(margin, height-3.8*cm, f"Empresa: {company_name}"); c.drawString(margin, height-4.3*cm, f"País de origen: {country}")
    yk = height-5.2*cm
    _draw_kpi(c, margin, yk, "Total Banco", f"${totals['total_bank']:,.0f}")
    _draw_kpi(c, margin+6.5*cm, yk, "Total Vita",  f"${totals['total_vita']:,.0f}")
    _draw_kpi(c, margin+13.0*cm, yk, "Ahorro",      f"${totals['total_sav']:,.0f} ({totals['sav_pct']:,.2f}%)")
    y = height-6.4*cm; c.setFont("Helvetica-Bold", 11)
    headers = ["Origen","Destino","Ruta","Monto (USD)","Banco","Vita","Ahorro"]; col_w = [3.0*cm,3.0*cm,3.0*cm,3.0*cm,2.5*cm,2.5*cm,2.5*cm]
    x = margin
    for h,w in zip(headers,col_w): c.drawString(x, y, h); x += w
    c.setLineWidth(0.5); c.line(margin, y-3, width-margin, y-3)
    c.setFont("Helvetica", 10); y -= 0.8*cm
    for _, r in df_cards.iterrows():
        x = margin
        row_vals = [r.get("Origen",""), r.get("Destino",""), r.get("Ruta Vita",""),
                    f"{r.get('Monto (USD)',0):,.0f}", f"{r.get('Costo Banco (USD)',0):,.0f}",
                    f"{r.get('Costo Vita (USD)',0):,.0f}", f"{r.get('Ahorro (USD)',0):,.0f}"]
        for val,w in zip(row_vals,col_w): c.drawString(x, y, str(val)); x += w
        y -= 0.7*cm
        if y < 3.0*cm: c.showPage(); y = height-2.5*cm; c.setFont("Helvetica", 10)
    c.showPage(); c.save(); buffer.seek(0); return buffer.read()

def send_email_with_attachment(sender, password, to, subject, body, pdf_bytes, filename, smtp_server, smtp_port, use_tls=True):
    msg = EmailMessage(); msg["From"]=sender; msg["To"]=to; msg["Subject"]=subject; msg.set_content(body or "")
    if pdf_bytes: msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)
    import smtplib
    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        if use_tls: server.starttls()
        server.login(sender, password)
        server.send_message(msg)

# =============================
# 2) Resultados
# =============================
with results_tab:
    st.subheader("Paso 2 – Resultados")
    if not st.session_state.flujos:
        st.info("Aún no has agregado flujos. Ve a la pestaña **Inputs**.")
    else:
        st.session_state.setdefault("cards_per_row", 3)
        st.session_state.cards_per_row = st.radio("Tarjetas por fila", [3,2], index=[3,2].index(st.session_state.cards_per_row), horizontal=True)
        df = pd.DataFrame(st.session_state.flujos)
        
        
        

        def calc_costos(row):
            monto = float(row["Monto (USD)"]); n = int(row["# Pagos"])
            
            # El cálculo del costo del banco se mantiene igual
            bank_cost = n*float(row["Banco fijo"]) + monto*(float(row["Banco %"])/100.0)
            
            # --- LÓGICA CORREGIDA PARA VITA ---
            # Si el tipo de operación es "Recepción", el costo base es cero.
            if row["Tipo"] == "Recepción":
                vita_cost = 0.0
            else: # Para "Envío", se usa el cálculo original
                vita_cost = n*float(row["Vita fijo"]) + monto*(float(row["Vita %"])/100.0)

            # --- SE AÑADE EL COSTO DE FX (SI APLICA) A AMBOS ---
            # Tanto el banco como Vita pueden aplicar un spread de FX.
            # Mantenemos este cálculo separado para ambos casos.
            if bool(row["FX aplica"]):
                bank_cost += monto*(float(row["FX Banco %"])/100.0)
                vita_cost += monto*(float(row["FX Vita %"])/100.0)
                
            ahorro = bank_cost - vita_cost
            ahorro_pct = (ahorro/bank_cost*100.0) if bank_cost>0 else 0.0
            
            return pd.Series({
                "Costo Banco (USD)": bank_cost, 
                "Costo Vita (USD)": vita_cost, 
                "Ahorro (USD)": ahorro, 
                "Ahorro %": ahorro_pct
            })

        def flag_code(name: str) -> str:
            mapping = {"estados unidos":"us","europa":"eu","reino unido":"gb","china":"cn","chile":"cl","perú":"pe","peru":"pe","colombia":"co","argentina":"ar","brasil":"br","méxico":"mx","mexico":"mx","uruguay":"uy"}
            return mapping.get((name or "").lower(), "us")

        def flag_img(code: str, size: int = 36):
            url = f"https://flagcdn.com/w{size}/{code}.png"; st.image(url, width=size)

        df_out = pd.concat([df, df.apply(calc_costos, axis=1)], axis=1)

        with st.container(border=True):
            cinfo, ck1, ck2, ck3, ck4 = st.columns([1.5,0.8,0.8,0.8,0.8])
            with cinfo:
                st.markdown(f"**Empresa:** {st.session_state.empresa_nombre}<br>**País de origen:** {st.session_state.empresa_pais}", unsafe_allow_html=True)
                if st.session_state.empresa_logo_bytes: st.image(st.session_state.empresa_logo_bytes, width=90)
            total_bank = float(df_out["Costo Banco (USD)"].sum()); total_vita = float(df_out["Costo Vita (USD)"].sum())
            total_sav = total_bank - total_vita; sav_pct = (total_sav/total_bank*100.0) if total_bank>0 else 0.0
            with ck1: st.metric("Total Banco", f"${total_bank:,.0f}")
            with ck2: st.metric("Total Vita", f"${total_vita:,.0f}")
            with ck3: st.metric("Total Ahorro", f"${total_sav:,.0f}")
            with ck4: st.metric("Ahorro % Total", f"{sav_pct:,.2f}%")

        st.markdown("### 💡 Ahorro por flujo (visual)")
        cols_per_row = int(st.session_state.cards_per_row); row_cols = None
        for i, row in df_out.iterrows():
            if i % cols_per_row == 0: row_cols = st.columns(cols_per_row)
            col = row_cols[i % cols_per_row]
            with col:
                with st.container(border=True):
                    hc1, hc2, hc3 = st.columns([0.25,0.5,0.25])
                    with hc1: flag_img(flag_code(row.get("Origen")), 40)
                    with hc2:
                        st.markdown(f"**{row.get('Origen','')} → {row.get('Destino','')}**")
                        subtitle = [f"Empresa: {st.session_state.empresa_nombre}", f"País: {st.session_state.empresa_pais}"]
                        if row.get("Ruta Vita"): subtitle.append(f"Ruta: {row['Ruta Vita']}")
                        if row.get("Tiempo Ruta"): subtitle.append(f"{row['Tiempo Ruta']}")
                        if row.get("Criterio Ruta"): subtitle.append(f"{row['Criterio Ruta']}")
                        st.caption(" · ".join(subtitle))
                    with hc3: flag_img(flag_code(row.get("Destino")), 40)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=["Banco"], y=[row["Costo Banco (USD)"]], name="Banco", width=[0.35]))
                    fig.add_trace(go.Bar(x=["Vita"],  y=[row["Costo Vita (USD)"]], name="Vita",  width=[0.35]))
                    fig.update_layout(barmode="group", xaxis_title="", yaxis_title="Costo (USD)", height=230, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Banco", f"${row['Costo Banco (USD)']:,.0f}")
                    m2.metric("Vita",  f"${row['Costo Vita (USD)']:,.0f}")
                    m3.metric("Ahorro", f"${row['Ahorro (USD)']:,.0f}")
                    m4.metric("Ahorro %", f"{row['Ahorro %']:.2f}%")

        st.markdown("---")
        with st.expander("Ver tabla completa (opcional)"):
            styled = (df_out[["Empresa","País de origen","Origen","Destino","Cuenta recepción","Ruta Vita","Tiempo Ruta","Criterio Ruta","# Pagos","Monto (USD)","Costo Banco (USD)","Costo Vita (USD)","Ahorro (USD)","Ahorro %"]]
                .style.format({"Monto (USD)":"{:,.0f}","Costo Banco (USD)":"{:,.0f}","Costo Vita (USD)":"{:,.0f}","Ahorro (USD)":"{:,.0f}","Ahorro %":"{:,.2f} %"}))
            st.dataframe(styled, use_container_width=True)

        # Exportar / Compartir
        totals = {"total_bank": float(df_out["Costo Banco (USD)"].sum()),
                  "total_vita": float(df_out["Costo Vita (USD)"].sum()),
                  "total_sav":  float(df_out["Costo Banco (USD)"].sum() - df_out["Costo Vita (USD)"].sum()),
                  "sav_pct":    float(((df_out["Costo Banco (USD)"].sum()-df_out["Costo Vita (USD)"].sum())/df_out["Costo Banco (USD)"].sum()*100.0) if df_out["Costo Banco (USD)"].sum()>0 else 0.0)}
        pdf_bytes = build_pdf(st.session_state.empresa_nombre, st.session_state.empresa_pais, st.session_state.empresa_logo_bytes, totals,
                              df_out[["Origen","Destino","Ruta Vita","Monto (USD)","Costo Banco (USD)","Costo Vita (USD)","Ahorro (USD)"]])
        file_name_pdf = f"Comparador_{st.session_state.empresa_nombre.replace(' ','_')}.pdf"
        st.download_button("⬇️ Descargar PDF", data=pdf_bytes, file_name=file_name_pdf, mime="application/pdf", use_container_width=True)

        with st.expander("Enviar por correo (adjunta el PDF)"):
            c_to, c_subj = st.columns([1.2, 1.0])
            with c_to: to_email = st.text_input("Para (email)", help="Ej: cliente@empresa.com")
            with c_subj: subject = st.text_input("Asunto", value=f"Comparador de Pagos – {st.session_state.empresa_nombre}")
            body = st.text_area("Mensaje", value=(f"Hola,\n\nAdjunto el comparador de pagos para {st.session_state.empresa_nombre} ({st.session_state.empresa_pais}).\n\nSaludos,\n"))
            smtp_server = st.text_input("Servidor SMTP", value="smtp.gmail.com")
            smtp_port = st.number_input("Puerto SMTP", value=587, step=1)
            sender_email = st.text_input("Tu correo (remitente)")
            sender_pass = st.text_input("Tu contraseña/app password", type="password")
            if st.button("✉️ Enviar correo ahora", use_container_width=True):
                if not (to_email and subject and sender_email and sender_pass and smtp_server and smtp_port):
                    st.error("Completa Para, Asunto, remitente, contraseña y SMTP.")
                else:
                    try:
                        send_email_with_attachment(sender_email, sender_pass, to_email, subject, body, pdf_bytes, file_name_pdf, smtp_server, int(smtp_port), True)
                        st.success("Correo enviado correctamente.")
                    except Exception as e:
                        st.error(f"No se pudo enviar el correo: {e}")
