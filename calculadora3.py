# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 21:10:46 2025
"""
# streamlit_app.py
# -------------------------------------------------------------
# Calculadora de Ahorro (simple): Banco vs Vita Wallet
# Autor: ChatGPT (para Pablo)
# Descripción: Versión simple + spread FX global.
# -------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Calculadora de Ahorro – Banco vs Vita Wallet", layout="wide")

# =============================
# 0) ENCABEZADO
# =============================
st.title("💸 Calculadora de Ahorro – Banco vs Vita Wallet")
st.caption("Versión simple con rieles por país y **spread FX** global (banco vs Vita).")

# =============================
# 1) PRESETS (editar en sidebar)
# =============================
with st.sidebar:
    st.header("⚙️ Parámetros Globales")
    total_ventas_usd = st.number_input(
        "Ventas anuales (USD)", min_value=0.0, value=25_000_000.0, step=100_000.0, format="%.2f"
    )

    st.markdown("---")
    st.subheader("Presets de comisiones (por transacción)")
    st.caption("Valores base por riel. Cada país hereda estos valores (puedes cambiarlos luego si lo necesitas en el código).")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Banca tradicional**")
        bank_swift_wire_fixed = st.number_input("SWIFT/Wire (USD)", min_value=0.0, value=50.0, step=1.0, key="bank_swift_wire_fixed")
        bank_sepa_std_fixed = st.number_input("SEPA estándar (EUR)", min_value=0.0, value=20.0, step=1.0, key="bank_sepa_std_fixed")
        bank_sepa_inst_fixed = st.number_input("SEPA instantáneo (EUR)", min_value=0.0, value=30.0, step=1.0, key="bank_sepa_inst_fixed")
        bank_local_fixed = st.number_input("Pago local/TEF (equivalente USD)", min_value=0.0, value=15.0, step=1.0, key="bank_local_fixed")
        bank_ach_fixed = st.number_input("ACH (USD)", min_value=0.0, value=0.5, step=0.5, key="bank_ach_fixed")
        bank_stablecoin_fixed = st.number_input("Stablecoin (USD)", min_value=0.0, value=0.0, step=0.0, key="bank_stablecoin_fixed")
        bank_pct_fee = st.number_input("% comisión (sobre monto)", min_value=0.0, value=0.5, step=0.05, format="%.2f", key="bank_pct_fee") / 100.0
        bank_fx_spread = st.number_input("% spread FX banco (sobre monto)", min_value=0.0, value=0.20, step=0.05, format="%.2f", key="bank_fx_spread") / 100.0

    with col_b:
        st.markdown("**Cargo FX by Vita**")
        vita_wire_usd_fixed = st.number_input("SWIFT/Wire (USD)", min_value=0.0, value=20.0, step=1.0, key="vita_wire_usd_fixed")
        vita_sepa_std_fixed = st.number_input("SEPA estándar (EUR)", min_value=0.0, value=20.0, step=1.0, key="vita_sepa_std_fixed")
        vita_sepa_inst_fixed = st.number_input("SEPA instantáneo (EUR)", min_value=0.0, value=20.0, step=1.0, key="vita_sepa_inst_fixed")
        vita_local_fixed = st.number_input("Pago local/TEF (equivalente USD)", min_value=0.0, value=5.0, step=1.0, key="vita_local_fixed")
        vita_ach_fixed = st.number_input("ACH (USD)", min_value=0.0, value=0.0, step=0.5, key="vita_ach_fixed")
        vita_stablecoin_fixed = st.number_input("Stablecoin (USD)", min_value=0.0, value=3.0, step=1.0, key="vita_stablecoin_fixed")
        vita_pct_fee = st.number_input("% comisión (sobre monto)", min_value=0.0, value=0.0, step=0.05, format="%.2f", key="vita_pct_fee") / 100.0
        vita_fx_spread = st.number_input("% spread FX Vita (sobre monto)", min_value=0.0, value=0.20, step=0.05, format="%.2f", key="vita_fx_spread") / 100.0

    st.markdown("---")
    optimize_routes = st.checkbox("Optimizar automáticamente la ruta Vita (mínimo costo)", value=True)
    allow_stablecoin_china = st.checkbox("Permitir stablecoins para China", value=True)
    st.caption("Si está activo, el modelo elegirá el riel Vita más barato permitido por país.")

# Diccionarios de presets por riel
BANK_PRESETS = {
    "SWIFT_WIRE": {"fixed": bank_swift_wire_fixed, "pct": bank_pct_fee},
    "SEPA_STD": {"fixed": bank_sepa_std_fixed, "pct": bank_pct_fee},
    "SEPA_INST": {"fixed": bank_sepa_inst_fixed, "pct": bank_pct_fee},
    "ACH": {"fixed": bank_ach_fixed, "pct": bank_pct_fee},
    "LOCAL": {"fixed": bank_local_fixed, "pct": bank_pct_fee},
}

VITA_PRESETS = {
    "WIRE_USD": {"fixed": vita_wire_usd_fixed, "pct": vita_pct_fee},
    "SWIFT_USD": {"fixed": vita_wire_usd_fixed, "pct": vita_pct_fee},
    "SEPA_STD": {"fixed": vita_sepa_std_fixed, "pct": vita_pct_fee},
    "SEPA_INST": {"fixed": vita_sepa_inst_fixed, "pct": vita_pct_fee},
    "ACH_US": {"fixed": vita_ach_fixed, "pct": vita_pct_fee},
    "LOCAL": {"fixed": vita_local_fixed, "pct": vita_pct_fee},
    "STABLECOIN": {"fixed": vita_stablecoin_fixed, "pct": 0.0},
}

# =============================
# 2) ESCENARIO BASE (editable)
# =============================
base_rows = [
    {"Pais": "Estados Unidos", "Moneda": "USD", "Origen": "Cuenta en EE.UU.", "Monto_USD": 4_000_000, "Ticket_promedio_USD": 50_000,
     "Pagos": None, "Banco_riel": "SWIFT_WIRE", "Vita_riel": "ACH_US", "Permitir_stablecoin": False, "FX_aplica": False},
    {"Pais": "Europa (EUR)", "Moneda": "EUR", "Origen": "Europa", "Monto_USD": 5_000_000, "Ticket_promedio_USD": 40_000,
     "Pagos": None, "Banco_riel": "SEPA_STD", "Vita_riel": "SEPA_STD", "Permitir_stablecoin": False, "FX_aplica": True},
    {"Pais": "China", "Moneda": "USD", "Origen": "China o cuenta USD en EE.UU.", "Monto_USD": 10_000_000, "Ticket_promedio_USD": 80_000,
     "Pagos": None, "Banco_riel": "SWIFT_WIRE", "Vita_riel": "SWIFT_USD", "Permitir_stablecoin": True, "FX_aplica": False},
    {"Pais": "México (USD)", "Moneda": "USD", "Origen": "México o cuenta USD en EE.UU.", "Monto_USD": 2_000_000, "Ticket_promedio_USD": 40_000,
     "Pagos": None, "Banco_riel": "SWIFT_WIRE", "Vita_riel": "SWIFT_USD", "Permitir_stablecoin": False, "FX_aplica": False},
    {"Pais": "Brasil (USD)", "Moneda": "USD", "Origen": "Brasil", "Monto_USD": 1_500_000, "Ticket_promedio_USD": 30_000,
     "Pagos": None, "Banco_riel": "SWIFT_WIRE", "Vita_riel": "SWIFT_USD", "Permitir_stablecoin": False, "FX_aplica": False},
    {"Pais": "Colombia (USD)", "Moneda": "USD", "Origen": "Colombia", "Monto_USD": 1_000_000, "Ticket_promedio_USD": 25_000,
     "Pagos": None, "Banco_riel": "SWIFT_WIRE", "Vita_riel": "SWIFT_USD", "Permitir_stablecoin": False, "FX_aplica": False},
    {"Pais": "LatAm (10 países)", "Moneda": "USD", "Origen": "Diversos", "Monto_USD": 1_500_000, "Ticket_promedio_USD": 20_000,
     "Pagos": None, "Banco_riel": "SWIFT_WIRE", "Vita_riel": "SWIFT_USD", "Permitir_stablecoin": False, "FX_aplica": False},
]

base_df = pd.DataFrame(base_rows)

st.markdown("### ✏️ Escenario por país (edita directamente)")
st.caption("Puedes cambiar montos, ticket promedio, rieles, marcar si **aplica FX** y añadir filas.")

edited_df = st.data_editor(
    base_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Pais": st.column_config.TextColumn("País/Región"),
        "Moneda": st.column_config.SelectboxColumn("Moneda", options=["USD", "EUR"], help="Moneda que envía el cliente"),
        "Origen": st.column_config.TextColumn("Origen del pago"),
        "Monto_USD": st.column_config.NumberColumn("Monto anual (USD)", min_value=0.0, step=1000.0, format="%.0f"),
        "Ticket_promedio_USD": st.column_config.NumberColumn("Ticket promedio (USD)", min_value=1.0, step=1000.0, format="%.0f"),
        "Pagos": st.column_config.NumberColumn("# Pagos (opcional)", min_value=0, step=1, help="Si lo dejas vacío, se calcula como Monto/Ticket"),
        "Banco_riel": st.column_config.SelectboxColumn("Riel Banco", options=list(BANK_PRESETS.keys())),
        "Vita_riel": st.column_config.SelectboxColumn("Riel Vita", options=list(VITA_PRESETS.keys())),
        "Permitir_stablecoin": st.column_config.CheckboxColumn("Stablecoin permitido"),
        "FX_aplica": st.column_config.CheckboxColumn("¿Aplica FX a este flujo?"),
    },
    hide_index=True,
)

# Aviso de desbalance con ventas
sum_montos = float(edited_df["Monto_USD"].fillna(0).sum())
if sum_montos > 0 and total_ventas_usd > 0 and abs(sum_montos - total_ventas_usd) > 1:
    st.info(
        f"La suma por país es **${sum_montos:,.0f}** y las ventas anuales definidas son **${total_ventas_usd:,.0f}**. "
        "No es obligatorio que coincidan; ajusta si lo deseas."
    )

# =============================
# 3) FUNCIONES
# =============================

def compute_n_payments(row):
    pagos = row.get("Pagos")
    if pagos is None or (isinstance(pagos, float) and math.isnan(pagos)) or pagos == 0:
        ticket = max(1.0, float(row.get("Ticket_promedio_USD") or 1.0))
        monto = float(row.get("Monto_USD") or 0.0)
        return int(math.ceil(monto / ticket)) if monto > 0 else 0
    return int(pagos)


def cost_for(method_preset: dict, amount_usd: float, n_payments: int) -> float:
    fixed = float(method_preset.get("fixed", 0.0))
    pct = float(method_preset.get("pct", 0.0))
    return n_payments * fixed + amount_usd * pct


def allowed_vita_methods(row) -> list:
    moneda = row.get("Moneda", "USD")
    pais = (row.get("Pais") or "").lower()
    allow = ["WIRE_USD", "SWIFT_USD", "ACH_US", "LOCAL"]
    if moneda == "EUR" or "europa" in pais:
        allow += ["SEPA_STD", "SEPA_INST"]
    if allow_stablecoin_china and bool(row.get("Permitir_stablecoin", False)):
        allow += ["STABLECOIN"]
    return list(dict.fromkeys(allow))


def choose_best_vita_route(row, amount_usd: float, n_payments: int) -> str:
    methods = allowed_vita_methods(row)
    best_m, best_cost = None, float("inf")
    for m in methods:
        preset = VITA_PRESETS.get(m, {"fixed": 9999, "pct": 1.0})
        c = cost_for(preset, amount_usd, n_payments)
        if c < best_cost:
            best_cost, best_m = c, m
    return best_m or row.get("Vita_riel", "WIRE_USD")

# =============================
# 4) CÁLCULOS
# =============================
rows = []
for _, r in edited_df.iterrows():
    amount = float(r.get("Monto_USD") or 0.0)
    n = compute_n_payments(r)

    # Banco
    bank_method = str(r.get("Banco_riel") or "SWIFT_WIRE").strip()
    bank_preset = BANK_PRESETS.get(bank_method, {"fixed": 0.0, "pct": 0.0})
    bank_cost = cost_for(bank_preset, amount, n)

    # Vita
    vita_method = str(r.get("Vita_riel") or "WIRE_USD").strip()
    if optimize_routes:
        vita_method = choose_best_vita_route(r, amount, n)
    vita_preset = VITA_PRESETS.get(vita_method, {"fixed": 0.0, "pct": 0.0})
    vita_cost = cost_for(vita_preset, amount, n)

    # FX spread (si aplica)
    fx_aplica = bool(r.get("FX_aplica", False))
    bank_fx_cost = amount * bank_fx_spread if fx_aplica else 0.0
    vita_fx_cost = amount * vita_fx_spread if fx_aplica else 0.0

    bank_total = bank_cost + bank_fx_cost
    vita_total = vita_cost + vita_fx_cost

    savings = bank_total - vita_total
    savings_pct = (savings / bank_total * 100.0) if bank_total > 0 else 0.0

    rows.append({
        "País/Región": r.get("Pais"),
        "Moneda": r.get("Moneda"),
        "Riel Banco": bank_method,
        "Riel Vita": vita_method,
        "Monto (USD)": amount,
        "# Pagos": n,
        "% FX Banco": bank_fx_spread * 100.0 if fx_aplica else 0.0,
        "% FX Vita": vita_fx_spread * 100.0 if fx_aplica else 0.0,
        "Costo Banco (USD)": bank_total,
        "Costo Vita (USD)": vita_total,
        "Ahorro (USD)": savings,
        "Ahorro %": savings_pct,
    })

result_df = pd.DataFrame(rows)

# =============================
# 5) KPIs
# =============================
col1, col2, col3, col4 = st.columns(4)

total_bank = float(result_df["Costo Banco (USD)"].sum()) if not result_df.empty else 0.0
total_vita = float(result_df["Costo Vita (USD)"].sum()) if not result_df.empty else 0.0
total_savings = total_bank - total_vita
savings_pct_total = (total_savings / total_bank * 100.0) if total_bank > 0 else 0.0

with col1:
    st.metric("Costo total – Banco", f"${total_bank:,.0f}")
with col2:
    st.metric("Costo total – Vita", f"${total_vita:,.0f}")
with col3:
    st.metric("Ahorro total", f"${total_savings:,.0f}")
with col4:
    st.metric("Ahorro %", f"{savings_pct_total:,.2f}%")

# =============================
# 6) TABLA
# =============================
st.markdown("### 📊 Resultados por país")
st.dataframe(result_df.style.format({
    "Monto (USD)": "{:,.0f}",
    "% FX Banco": "{:,.2f}%",
    "% FX Vita": "{:,.2f}%",
    "Costo Banco (USD)": "{:,.0f}",
    "Costo Vita (USD)": "{:,.0f}",
    "Ahorro (USD)": "{:,.0f}",
    "Ahorro %": "{:,.2f}%",
}), use_container_width=True)

# =============================
# 7) GRÁFICO
# =============================
st.markdown("### 📈 Costos por país: Banco vs Vita (con FX si aplica)")
if not result_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=result_df["País/Región"], y=result_df["Costo Banco (USD)"], name="Banco"))
    fig.add_trace(go.Bar(x=result_df["País/Región"], y=result_df["Costo Vita (USD)"], name="Vita Wallet"))
    fig.update_layout(barmode="group", xaxis_title="País/Región", yaxis_title="Costo (USD)", height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Agrega países y montos en la tabla para ver el gráfico.")

# =============================
# 8) DESCARGA CSV
# =============================
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

csv_bytes = convert_df_to_csv(result_df)
st.download_button("⬇️ Descargar resultados (CSV)", data=csv_bytes, file_name="resultados_ahorro_vita.csv", mime="text/csv")

st.markdown("---")
st.caption(
    """
    Notas:
    • El **spread FX** se aplica sólo a las filas marcadas con "¿Aplica FX a este flujo?".
    • Por defecto asumimos USD para casi todos los clientes y EUR para Europa.
    • Los valores de comisiones y spreads son referenciales y se pueden ajustar.
    """
)
