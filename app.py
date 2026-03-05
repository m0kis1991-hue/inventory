import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Page
st.set_page_config(page_title="InventoryAI", page_icon="📦", layout="wide")
st.title("📦 InventoryAI")
st.caption("Upload sales CSV → Forecast demand → Reorder suggestions (SaaS-ready MVP)")

# -------------------------
# Helpers

@st.cache_data
def generate_sample_data(num_skus: int = 10, days: int = 365) -> pd.DataFrame:
    np.random.seed(42)
    skus = [f"SKU{i:03}" for i in range(1, num_skus + 1)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rows = []
    for sku in skus:
        base_lam = np.random.randint(5, 40)
        stock = np.random.randint(200, 600)

        for d in dates:
            qty = np.random.poisson(lam=base_lam)
            replen = np.random.randint(0, 15)

            stock = stock - qty + replen
            if stock < 0:
                stock = np.random.randint(50, 250)

            rows.append([d.strftime("%Y-%m-%d"), sku, int(qty), int(stock)])

    return pd.DataFrame(rows, columns=["Date", "SKU", "Quantity", "Stock"])


def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Date", "SKU", "Quantity", "Stock"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"❌ SALES CSV: λείπουν στήλες: {', '.join(sorted(missing))}")
        st.info("✅ Απαιτούνται: Date, SKU, Quantity, Stock")
        st.stop()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        st.error("❌ SALES CSV: η στήλη Date έχει λάθος μορφή ημερομηνίας.")
        st.info("Παράδειγμα: 2026-03-04")
        st.stop()

    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0)

    # sort for correct last stock
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    return df


def validate_products(df: pd.DataFrame) -> pd.DataFrame:
    # required SKU only; others optional
    if "SKU" not in df.columns:
        st.warning("⚠️ PRODUCTS CSV δεν έχει στήλη 'SKU'. Θα αγνοηθεί.")
        return None

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # normalize optional fields if present
    for col in ["ProductName", "Category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    # per-SKU params (optional)
    num_cols = ["LeadTime", "DaysToCover", "MinStock", "ServiceLevel"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def z_from_service(service_level: float) -> float:
    z_table = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 99: 2.33}
    # service_level can be float; we map to nearest key
    keys = np.array(list(z_table.keys()))
    nearest = int(keys[np.argmin(np.abs(keys - service_level))])
    return float(z_table[nearest])


def safe_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# -------------------------
# Sidebar - Global defaults
st.sidebar.header("⚙️ Global defaults")

use_sample = st.sidebar.checkbox("✅ Use demo sample data", value=True)

default_lead_time = st.sidebar.number_input("Default Lead Time (days)", min_value=1, value=7)
default_service_level = st.sidebar.slider("Default Service Level (%)", min_value=80, max_value=99, value=95)
default_days_to_cover = st.sidebar.number_input("Default Days to Cover", min_value=1, value=14)
default_min_stock = st.sidebar.number_input("Default Min Stock (optional rule)", min_value=0, value=0)

st.sidebar.divider()
st.sidebar.subheader("📁 Uploads")

uploaded_sales = None
uploaded_products = None
if not use_sample:
    uploaded_sales = st.sidebar.file_uploader("Upload SALES CSV", type="csv")
    uploaded_products = st.sidebar.file_uploader("Upload PRODUCTS CSV (optional)", type="csv")

# -------------------------
# Load data
products_df = None

if use_sample:
    st.info("ℹ️ Demo mode: using sample data (10 SKUs / 365 days).")
    sales_df = generate_sample_data(num_skus=10, days=365)
    sales_df = validate_sales(sales_df)
else:
    if uploaded_sales is None:
        st.warning("⬅️ Ανέβασε SALES CSV για να ξεκινήσεις.")
        st.stop()

    sales_df = pd.read_csv(uploaded_sales)
    sales_df = validate_sales(sales_df)
    st.success("✅ SALES CSV loaded!")

    if uploaded_products is not None:
        products_df = pd.read_csv(uploaded_products)
        products_df = validate_products(products_df)
        if products_df is not None:
            st.success("✅ PRODUCTS CSV loaded!")

# -------------------------
# Session-state product catalog (manual manager)
if "catalog" not in st.session_state:
    # seed from products CSV if exists; else empty
    if products_df is not None:
        st.session_state["catalog"] = products_df.copy()
    else:
        st.session_state["catalog"] = pd.DataFrame(
            columns=["SKU", "ProductName", "Category", "LeadTime", "DaysToCover", "ServiceLevel", "MinStock"]
        )

catalog = st.session_state["catalog"]

# -------------------------
# Navigation
tab_products, tab_forecast, tab_help = st.tabs(["📦 Products", "📊 Forecast", "❓ Help"])

# =========================
# TAB: PRODUCTS
with tab_products:
    st.subheader("📦 Products Manager (optional)")
    st.caption("Here you can keep a simple product catalog + per-SKU parameters. "
               "You can also upload a PRODUCTS CSV, and it will prefill this table.")

    with st.expander("➕ Add / Update a product", expanded=True):
        c1, c2, c3 = st.columns(3)
        sku_in = c1.text_input("SKU (required)", value="")
        name_in = c2.text_input("ProductName (optional)", value="")
        cat_in = c3.text_input("Category (optional)", value="")

        c4, c5, c6, c7 = st.columns(4)
        lead_in = c4.number_input("LeadTime override (days)", min_value=0, value=0)
        cover_in = c5.number_input("DaysToCover override", min_value=0, value=0)
        sl_in = c6.selectbox("ServiceLevel override (%)", options=[0, 80, 85, 90, 95, 99], index=0)
        min_in = c7.number_input("MinStock override", min_value=0, value=0)

        colA, colB = st.columns(2)
        if colA.button("💾 Save product"):
            sku_clean = sku_in.strip()
            if not sku_clean:
                st.error("SKU είναι υποχρεωτικό.")
            else:
                row = {
                    "SKU": sku_clean,
                    "ProductName": name_in.strip(),
                    "Category": cat_in.strip(),
                    "LeadTime": lead_in if lead_in > 0 else np.nan,
                    "DaysToCover": cover_in if cover_in > 0 else np.nan,
                    "ServiceLevel": sl_in if sl_in > 0 else np.nan,
                    "MinStock": min_in if min_in > 0 else np.nan,
                }

                if "SKU" in catalog.columns and (catalog["SKU"] == sku_clean).any():
                    idx = catalog.index[catalog["SKU"] == sku_clean][0]
                    for k, v in row.items():
                        catalog.at[idx, k] = v
                else:
                    catalog = pd.concat([catalog, pd.DataFrame([row])], ignore_index=True)

                st.session_state["catalog"] = catalog
                st.success(f"Saved: {sku_clean}")

        if colB.button("🧹 Clear catalog (demo only)"):
            st.session_state["catalog"] = pd.DataFrame(
                columns=["SKU", "ProductName", "Category", "LeadTime", "DaysToCover", "ServiceLevel", "MinStock"]
            )
            catalog = st.session_state["catalog"]
            st.warning("Catalog cleared.")

    st.markdown("### Current catalog")
    st.dataframe(st.session_state["catalog"], use_container_width=True)

    # Download catalog template
    st.markdown("### Download PRODUCTS template")
    template = pd.DataFrame(
        [{
            "SKU": "SKU001",
            "ProductName": "Example product",
            "Category": "Example category",
            "LeadTime": 7,
            "DaysToCover": 14,
            "ServiceLevel": 95,
            "MinStock": 0,
        }]
    )
    st.download_button(
        "⬇️ Download products_template.csv",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="products_template.csv",
        mime="text/csv"
    )

# =========================
# TAB: FORECAST
with tab_forecast:
    st.subheader("📊 Forecast & Reorder Suggestions")
    st.caption("Sales data is grouped by SKU and calculated with either global defaults or per-SKU overrides.")

    st.markdown("### Sales data preview")
    st.dataframe(sales_df.head(20), use_container_width=True)

    # Merge catalog onto results later
    catalog = st.session_state["catalog"].copy()
    if not catalog.empty:
        catalog["SKU"] = catalog["SKU"].astype(str).str.strip()

    # Forecast engine
    results = []
    for sku in sales_df["SKU"].unique():
        temp = sales_df[sales_df["SKU"] == sku].copy()

        avg_demand = float(temp["Quantity"].mean())
        std_demand = float(temp["Quantity"].std(ddof=1)) if len(temp) > 1 else 0.0
        current_stock = float(temp["Stock"].iloc[-1])

        # per-SKU overrides (if in catalog)
        lead_time = default_lead_time
        days_to_cover = default_days_to_cover
        service_level = default_service_level
        min_stock = default_min_stock

        if not catalog.empty and (catalog["SKU"] == sku).any():
            row = catalog[catalog["SKU"] == sku].iloc[0]
            lead_time = safe_int(row.get("LeadTime"), lead_time) or lead_time
            days_to_cover = safe_int(row.get("DaysToCover"), days_to_cover) or days_to_cover
            service_level = safe_int(row.get("ServiceLevel"), service_level) or service_level
            min_stock = safe_float(row.get("MinStock"), min_stock) if default_min_stock > 0 else safe_float(row.get("MinStock"), min_stock)

        Z = z_from_service(float(service_level))

        safety_stock = float(Z * std_demand * np.sqrt(lead_time))
        reorder_point = float(avg_demand * lead_time + safety_stock)

        target_stock = float(avg_demand * days_to_cover + safety_stock)
        # Optional rule: ensure target stock >= min_stock
        if min_stock and min_stock > 0:
            target_stock = max(target_stock, float(min_stock))

        order_qty = max(0, int(round(target_stock - current_stock)))
        status = "✅ OK" if current_stock >= reorder_point else "⚠️ LOW STOCK"

        results.append({
            "SKU": sku,
            "Avg Demand": round(avg_demand, 2),
            "Std Demand": round(std_demand, 2),
            "LeadTime": int(lead_time),
            "ServiceLevel": int(service_level),
            "DaysToCover": int(days_to_cover),
            "Safety Stock": round(safety_stock, 2),
            "Reorder Point": round(reorder_point, 2),
            "Target Stock": round(target_stock, 2),
            "Current Stock": int(round(current_stock)),
            "Order Qty": int(order_qty),
            "Status": status,
        })

    res_df = pd.DataFrame(results)

    # merge product fields if exist
    if not catalog.empty and "SKU" in catalog.columns:
        # keep only product info columns (avoid duplicate params)
        keep_cols = [c for c in ["SKU", "ProductName", "Category"] if c in catalog.columns]
        res_df = res_df.merge(catalog[keep_cols].drop_duplicates("SKU"), on="SKU", how="left")

    # Days of cover & severity
    res_df["Days of Cover"] = res_df.apply(
        lambda r: round(r["Current Stock"] / r["Avg Demand"], 1) if r["Avg Demand"] > 0 else np.nan,
        axis=1,
    )
    res_df["Severity"] = (res_df["Reorder Point"] - res_df["Current Stock"]).clip(lower=0)

    # Executive summary
    st.markdown("### 🧾 Executive Summary")
    total_skus = len(res_df)
    low_count = int(res_df["Status"].str.contains("LOW STOCK").sum())
    ok_count = int(total_skus - low_count)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", total_skus)
    c2.metric("LOW STOCK", low_count)
    c3.metric("OK", ok_count)
    c4.metric("Units to order", int(res_df["Order Qty"].sum()))

    critical = res_df.sort_values(["Severity", "Days of Cover"], ascending=[False, True]).head(5)
    st.markdown("**Top 5 most critical**")
    show_cols_crit = [c for c in ["SKU", "ProductName", "Current Stock", "Reorder Point", "Days of Cover", "Order Qty", "Status"] if c in critical.columns]
    st.dataframe(critical[show_cols_crit], use_container_width=True)

    # Table
    st.markdown("### 📊 Reorder Table")
    show_only_orders = st.checkbox("Show only SKUs with Order Qty > 0", value=True)
    table_df = res_df[res_df["Order Qty"] > 0].copy() if show_only_orders else res_df.copy()
    table_df = table_df.sort_values(["Status", "Severity", "Order Qty"], ascending=[True, False, False])

    base_cols = ["SKU", "ProductName", "Category", "Current Stock", "Days of Cover", "Reorder Point",
                 "Target Stock", "Order Qty", "Status", "LeadTime", "ServiceLevel", "DaysToCover"]
    cols = [c for c in base_cols if c in table_df.columns] + [c for c in table_df.columns if c not in base_cols]

    st.dataframe(table_df[cols].drop(columns=["Severity"], errors="ignore"), use_container_width=True)

    # Downloads
    st.markdown("### ⬇️ Downloads")
    csv_bytes = table_df.drop(columns=["Severity"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="reorder_suggestions.csv", mime="text/csv")

    # Excel export
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        table_df.drop(columns=["Severity"], errors="ignore").to_excel(writer, index=False, sheet_name="Reorder")
        critical.drop(columns=["Severity"], errors="ignore").to_excel(writer, index=False, sheet_name="Top5")
    st.download_button(
        "Download Excel (reorder_report.xlsx)",
        data=buffer.getvalue(),
        file_name="reorder_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Plot
    st.markdown("### 📈 Plot (7-day avg sales)")
    sku_selected = st.selectbox("Select SKU for plot:", options=sorted(sales_df["SKU"].unique()))
    temp = sales_df[sales_df["SKU"] == sku_selected].copy()
    temp["Sales_7d_avg"] = temp["Quantity"].rolling(7, min_periods=1).mean()

    reorder_line = float(res_df.loc[res_df["SKU"] == sku_selected, "Reorder Point"].iloc[0])

    plt.figure(figsize=(12, 4))
    plt.plot(temp["Date"], temp["Sales_7d_avg"], label="7-day avg sales")
    plt.axhline(y=reorder_line, linestyle="--", label="Reorder Point")
    plt.title(f"{sku_selected} - Demand (7-day avg) & Reorder Point")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# =========================
# TAB: HELP
with tab_help:
    st.subheader("❓ How to use")
    st.markdown(
        """
**Typical workflow:**
1. (Optional) Add your products in **Products** tab (SKU, name, category, per-SKU params).
2. Upload your **SALES CSV** (must contain: Date, SKU, Quantity, Stock).
3. Check the **Forecast** tab for:
   - low stock alerts
   - suggested order quantity
   - top critical SKUs
4. Download the reorder report (CSV or Excel).

**CSV format (sales):**
- Date: YYYY-MM-DD
- SKU: product code
- Quantity: units sold
- Stock: units in stock at end of day (or latest snapshot)

**PRODUCTS CSV template:**
Download it from the Products tab.
        """
    )
