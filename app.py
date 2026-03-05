import sqlite3
from datetime import datetime
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="InventoryAI - Weekly CSV MVP", layout="wide")
st.title("📦 InventoryAI — Weekly CSV Inventory & Demand Forecasting (MVP)")

# -----------------------
# DB helpers (SQLite)
# -----------------------
DB_PATH = "inventory_ai.sqlite"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # SALES: daily totals by Date+SKU (unique)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sales (
            date TEXT NOT NULL,
            sku TEXT NOT NULL,
            quantity REAL NOT NULL,
            stock REAL NOT NULL,
            PRIMARY KEY (date, sku)
        )
        """
    )

    # PRODUCTS: optional catalog
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            sku TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            lead_time_override INTEGER,
            days_to_cover_override INTEGER,
            service_level_override INTEGER,
            min_stock_override REAL
        )
        """
    )

    conn.commit()
    conn.close()


def upsert_sales(df: pd.DataFrame) -> int:
    """Insert/replace sales rows; returns rows upserted."""
    conn = get_conn()
    cur = conn.cursor()

    rows = df[["Date", "SKU", "Quantity", "Stock"]].copy()
    rows["Date"] = rows["Date"].dt.strftime("%Y-%m-%d")
    rows["SKU"] = rows["SKU"].astype(str).str.strip()

    payload = list(rows.itertuples(index=False, name=None))

    cur.executemany(
        """
        INSERT OR REPLACE INTO sales (date, sku, quantity, stock)
        VALUES (?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()
    conn.close()
    return len(payload)


def upsert_products(df: pd.DataFrame) -> int:
    """Insert/replace products rows; returns rows upserted."""
    conn = get_conn()
    cur = conn.cursor()

    # Ensure required column
    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # Normalize optional columns
    def col_or_none(c):
        return df[c] if c in df.columns else None

    payload = []
    for _, r in df.iterrows():
        payload.append(
            (
                r["SKU"],
                r.get("ProductName", None),
                r.get("Category", None),
                int(r["LeadTime"]) if "LeadTime" in df.columns and pd.notna(r["LeadTime"]) else None,
                int(r["DaysToCover"]) if "DaysToCover" in df.columns and pd.notna(r["DaysToCover"]) else None,
                int(r["ServiceLevel"]) if "ServiceLevel" in df.columns and pd.notna(r["ServiceLevel"]) else None,
                float(r["MinStock"]) if "MinStock" in df.columns and pd.notna(r["MinStock"]) else None,
            )
        )

    cur.executemany(
        """
        INSERT OR REPLACE INTO products
        (sku, product_name, category, lead_time_override, days_to_cover_override, service_level_override, min_stock_override)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()
    conn.close()
    return len(payload)


def load_sales() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT date as Date, sku as SKU, quantity as Quantity, stock as Stock FROM sales", conn)
    conn.close()
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0.0)
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    return df


def load_products() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            sku as SKU,
            product_name as ProductName,
            category as Category,
            lead_time_override as LeadTime,
            days_to_cover_override as DaysToCover,
            service_level_override as ServiceLevel,
            min_stock_override as MinStock
        FROM products
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["SKU"] = df["SKU"].astype(str).str.strip()
    return df


def reset_db(confirm: bool):
    if not confirm:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sales")
    cur.execute("DELETE FROM products")
    conn.commit()
    conn.close()


# -----------------------
# Validation helpers
# -----------------------
def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Date", "SKU", "Quantity", "Stock"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"❌ SALES CSV λείπουν στήλες: {', '.join(sorted(missing))}")
        st.info("✅ Απαιτούνται ακριβώς: Date, SKU, Quantity, Stock")
        st.stop()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        st.error("❌ Η στήλη Date έχει τιμές που δεν αναγνωρίζονται ως ημερομηνία.")
        st.info("Παράδειγμα: 2026-03-04")
        st.stop()

    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0.0)

    # Daily totals mode: enforce unique Date+SKU by summing Quantity and taking last Stock (max date ties not needed)
    # If duplicates exist in the upload, we consolidate within the upload.
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    df = (
        df.groupby(["Date", "SKU"], as_index=False)
        .agg({"Quantity": "sum", "Stock": "last"})
    )

    return df


def validate_products(df: pd.DataFrame) -> pd.DataFrame:
    if "SKU" not in df.columns:
        st.error("❌ PRODUCTS CSV πρέπει να έχει στήλη SKU.")
        st.stop()

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # Optional columns normalization (if present)
    rename_map = {}
    # allow common variants
    for col in df.columns:
        c = col.strip()
        rename_map[col] = c
    df = df.rename(columns=rename_map)

    # expected optional names: ProductName, Category, LeadTime, DaysToCover, ServiceLevel, MinStock
    return df


def z_from_service_level(sl: int) -> float:
    # common approximations
    z_table = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 97: 1.88, 98: 2.05, 99: 2.33}
    return z_table.get(int(sl), 1.65)


def template_sales_csv() -> bytes:
    sample = pd.DataFrame(
        [
            ["2026-03-01", "SKU001", 5, 120],
            ["2026-03-01", "SKU002", 2, 80],
            ["2026-03-02", "SKU001", 4, 118],
        ],
        columns=["Date", "SKU", "Quantity", "Stock"],
    )
    return sample.to_csv(index=False).encode("utf-8")


def template_products_csv() -> bytes:
    sample = pd.DataFrame(
        [
            ["SKU001", "Product A", "Category 1", 7, 14, 95, 0],
            ["SKU002", "Product B", "Category 2", "", "", "", ""],
        ],
        columns=["SKU", "ProductName", "Category", "LeadTime", "DaysToCover", "ServiceLevel", "MinStock"],
    )
    return sample.to_csv(index=False).encode("utf-8")


# -----------------------
# Init DB
# -----------------------
init_db()

# -----------------------
# Sidebar - Global Defaults + Uploads
# -----------------------
st.sidebar.header("⚙️ Global defaults")

use_demo = st.sidebar.checkbox("Use demo sample data", value=False)

default_lead_time = st.sidebar.number_input("Default Lead Time (days)", min_value=1, value=7)
default_service_level = st.sidebar.slider("Default Service Level (%)", min_value=80, max_value=99, value=95)
default_days_to_cover = st.sidebar.number_input("Default Days to Cover", min_value=1, value=14)
default_min_stock = st.sidebar.number_input("Default Min Stock (optional rule)", min_value=0.0, value=0.0)

st.sidebar.divider()
st.sidebar.subheader("📤 Uploads")

sales_file = st.sidebar.file_uploader("Upload SALES CSV (weekly)", type="csv")
products_file = st.sidebar.file_uploader("Upload PRODUCTS CSV (optional)", type="csv")

col_u1, col_u2 = st.sidebar.columns(2)
do_upload = col_u1.button("✅ Apply upload")
do_reset = col_u2.button("🧨 Reset all data")

if do_reset:
    st.sidebar.warning("⚠️ Αυτό θα σβήσει ΟΛΑ τα δεδομένα (sales + products).")
    confirm = st.sidebar.checkbox("Confirm reset")
    if confirm:
        reset_db(True)
        st.sidebar.success("✅ Reset completed. Refresh the page.")
        st.stop()

# -----------------------
# Tabs
# -----------------------
tab_products, tab_forecast, tab_help = st.tabs(["📁 Products", "📈 Forecast", "❓ Help"])

# -----------------------
# Demo data (optional)
# -----------------------
def generate_demo_sales(num_skus=10, days=365) -> pd.DataFrame:
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
            rows.append([d, sku, float(qty), float(stock)])

    df = pd.DataFrame(rows, columns=["Date", "SKU", "Quantity", "Stock"])
    df = df.groupby(["Date", "SKU"], as_index=False).agg({"Quantity": "sum", "Stock": "last"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def generate_demo_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SKU": [f"SKU{i:03}" for i in range(1, 11)],
            "ProductName": [f"Demo Product {i}" for i in range(1, 11)],
            "Category": ["Demo"] * 10,
            "LeadTime": [None] * 10,
            "DaysToCover": [None] * 10,
            "ServiceLevel": [None] * 10,
            "MinStock": [None] * 10,
        }
    )

# -----------------------
# Handle upload action
# -----------------------
if do_upload:
    if use_demo:
        demo_sales = generate_demo_sales()
        upsert_sales(demo_sales)
        demo_products = generate_demo_products()
        upsert_products(demo_products)
        st.success("✅ Demo data loaded into database!")
    else:
        if sales_file is None:
            st.error("❌ Πρέπει να ανεβάσεις SALES CSV.")
            st.stop()

        sales_df = pd.read_csv(sales_file)
        sales_df = validate_sales(sales_df)
        n_sales = upsert_sales(sales_df)

        if products_file is not None:
            prod_df = pd.read_csv(products_file)
            prod_df = validate_products(prod_df)
            n_prod = upsert_products(prod_df)
            st.success(f"✅ SALES uploaded: {n_sales} rows | PRODUCTS uploaded: {n_prod} rows")
        else:
            st.success(f"✅ SALES uploaded: {n_sales} rows")
    st.info("ℹ️ Αν δεν βλέπεις αλλαγές, πάτα Refresh (F5).")

# -----------------------
# Load from DB
# -----------------------
sales = load_sales()
products = load_products()

# -----------------------
# PRODUCTS tab
# -----------------------
with tab_products:
    st.subheader("🗂️ Product catalog (optional)")
    st.caption("Μπορείς να ανεβάσεις PRODUCTS CSV ή να δουλέψεις χωρίς αυτό. Χρήσιμο για ονομασίες/κατηγορίες και per-SKU overrides.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "⬇️ Download SALES template.csv",
            data=template_sales_csv(),
            file_name="sales_template.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download PRODUCTS template.csv",
            data=template_products_csv(),
            file_name="products_template.csv",
            mime="text/csv",
        )

    st.divider()

    if products.empty:
        st.warning("Δεν υπάρχει PRODUCTS catalog ακόμη. (Προαιρετικό)")
    else:
        st.dataframe(products, use_container_width=True)

    st.divider()
    st.subheader("➕ Quick add / update one product (optional)")

    with st.expander("Add / Update a product"):
        p1, p2, p3 = st.columns([1, 2, 2])
        sku_in = p1.text_input("SKU (required)")
        name_in = p2.text_input("ProductName (optional)")
        cat_in = p3.text_input("Category (optional)")

        o1, o2, o3, o4 = st.columns(4)
        lt = o1.number_input("LeadTime override (days)", min_value=0, value=0)
        dc = o2.number_input("DaysToCover override", min_value=0, value=0)
        sl = o3.number_input("ServiceLevel override (%)", min_value=0, max_value=99, value=0)
        ms = o4.number_input("MinStock override", min_value=0.0, value=0.0)

        if st.button("💾 Save product"):
            if not sku_in.strip():
                st.error("SKU είναι υποχρεωτικό.")
            else:
                df_one = pd.DataFrame(
                    [{
                        "SKU": sku_in.strip(),
                        "ProductName": name_in.strip() if name_in else None,
                        "Category": cat_in.strip() if cat_in else None,
                        "LeadTime": lt if lt > 0 else None,
                        "DaysToCover": dc if dc > 0 else None,
                        "ServiceLevel": sl if sl > 0 else None,
                        "MinStock": ms if ms > 0 else None,
                    }]
                )
                upsert_products(df_one)
                st.success("✅ Saved! Refresh the page.")
                st.stop()

# -----------------------
# FORECAST tab
# -----------------------
with tab_forecast:
    st.subheader("Forecast & Reorder Suggestions")
    if sales.empty:
        st.warning("Δεν υπάρχουν SALES δεδομένα ακόμα. Ανέβασε SALES CSV (weekly) ή πάτα 'Use demo sample data' και 'Apply upload'.")
        st.stop()

    st.caption("Το σύστημα περιμένει daily totals ανά Date+SKU. Κάθε εβδομάδα κάνεις upload νέο CSV και το ιστορικό ενημερώνεται (append/replace).")

    # Preview
    st.markdown("### 📄 Sales data preview")
    st.dataframe(sales.tail(50), use_container_width=True)

    # Merge product info (optional)
    if not products.empty:
        sales2 = sales.merge(products[["SKU", "ProductName", "Category"]], on="SKU", how="left")
    else:
        sales2 = sales.copy()

    # Compute per-SKU stats
    Z_default = z_from_service_level(int(default_service_level))

    # Per-SKU overrides dict
    overrides = {}
    if not products.empty:
        for _, r in products.iterrows():
            overrides[r["SKU"]] = {
                "LeadTime": int(r["LeadTime"]) if pd.notna(r.get("LeadTime")) else None,
                "DaysToCover": int(r["DaysToCover"]) if pd.notna(r.get("DaysToCover")) else None,
                "ServiceLevel": int(r["ServiceLevel"]) if pd.notna(r.get("ServiceLevel")) else None,
                "MinStock": float(r["MinStock"]) if pd.notna(r.get("MinStock")) else None,
            }

    results = []
    for sku in sorted(sales["SKU"].unique()):
        temp = sales[sales["SKU"] == sku].copy().sort_values("Date")

        avg_demand = float(temp["Quantity"].mean())
        std_demand = float(temp["Quantity"].std(ddof=1)) if len(temp) > 1 else 0.0
        current_stock = float(temp["Stock"].iloc[-1])

        # Apply overrides if exist
        lt = overrides.get(sku, {}).get("LeadTime") or int(default_lead_time)
        dc = overrides.get(sku, {}).get("DaysToCover") or int(default_days_to_cover)
        sl = overrides.get(sku, {}).get("ServiceLevel") or int(default_service_level)
        z = z_from_service_level(int(sl))
        min_stock_rule = overrides.get(sku, {}).get("MinStock")
        if min_stock_rule is None:
            min_stock_rule = float(default_min_stock)

        safety_stock = z * std_demand * np.sqrt(lt)
        reorder_point = (avg_demand * lt) + safety_stock
        target_stock = (avg_demand * dc) + safety_stock

        # Apply minimum stock floor if set
        if min_stock_rule and min_stock_rule > 0:
            reorder_point = max(reorder_point, min_stock_rule)
            target_stock = max(target_stock, min_stock_rule)

        order_qty = max(0, int(round(target_stock - current_stock)))
        status = "✅ OK" if current_stock >= reorder_point else "⚠️ LOW STOCK"

        results.append({
            "SKU": sku,
            "AvgDemand": round(avg_demand, 2),
            "StdDemand": round(std_demand, 2),
            "LeadTime": int(lt),
            "ServiceLevel": int(sl),
            "DaysToCover": int(dc),
            "SafetyStock": round(safety_stock, 2),
            "ReorderPoint": round(reorder_point, 2),
            "TargetStock": round(target_stock, 2),
            "CurrentStock": int(round(current_stock)),
            "OrderQty": order_qty,
            "Status": status
        })

    res = pd.DataFrame(results)

    # Add product info columns if available
    if not products.empty:
        res = res.merge(products[["SKU", "ProductName", "Category"]], on="SKU", how="left")

    # Days of Cover and Severity
    res["DaysOfCover"] = res.apply(lambda r: round(r["CurrentStock"] / r["AvgDemand"], 1) if r["AvgDemand"] > 0 else np.nan, axis=1)
    res["Severity"] = (res["ReorderPoint"] - res["CurrentStock"]).clip(lower=0)

    # Executive summary
    st.markdown("### 🧾 Executive Summary")
    total_skus = len(res)
    low_count = int(res["Status"].str.contains("LOW STOCK").sum())
    ok_count = total_skus - low_count

    a, b, c, d = st.columns(4)
    a.metric("Products", total_skus)
    b.metric("⚠️ LOW STOCK", low_count)
    c.metric("✅ OK", ok_count)
    d.metric("Default DaysToCover", int(default_days_to_cover))

    st.markdown("**Top 5 most critical**")
    critical = res.sort_values(["Severity", "DaysOfCover"], ascending=[False, True]).head(5)

    show_cols = ["SKU", "ProductName", "Category", "CurrentStock", "ReorderPoint", "DaysOfCover", "OrderQty", "Status"]
    show_cols = [c for c in show_cols if c in critical.columns]
    st.dataframe(critical[show_cols], use_container_width=True)

    st.divider()

    st.markdown("### 📊 Reorder table")
    show_only_orders = st.checkbox("Show only SKUs needing order (OrderQty > 0)", value=True)
    out = res[res["OrderQty"] > 0].copy() if show_only_orders else res.copy()
    out = out.sort_values(["Status", "Severity", "OrderQty"], ascending=[True, False, False])

    # Friendly columns ordering
    preferred = ["SKU", "ProductName", "Category", "CurrentStock", "DaysOfCover", "ReorderPoint", "TargetStock", "OrderQty", "LeadTime", "ServiceLevel", "DaysToCover", "Status"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    st.dataframe(out[cols], use_container_width=True)

    st.download_button(
        "⬇️ Download reorder_suggestions.csv",
        data=out[cols].to_csv(index=False).encode("utf-8"),
        file_name="reorder_suggestions.csv",
        mime="text/csv",
    )

    st.divider()

    st.markdown("### 📈 SKU chart (7-day avg sales)")
    sku_selected = st.selectbox("Pick SKU", options=sorted(sales["SKU"].unique()))
    temp = sales[sales["SKU"] == sku_selected].copy().sort_values("Date")
    temp["Sales_7d_avg"] = temp["Quantity"].rolling(7, min_periods=1).mean()

    reorder_line = float(res.loc[res["SKU"] == sku_selected, "ReorderPoint"].iloc[0])

    plt.figure(figsize=(12, 4))
    plt.plot(temp["Date"], temp["Sales_7d_avg"], label="7-day avg sales")
    plt.axhline(y=reorder_line, linestyle="--", label="Reorder Point")
    plt.title(f"{sku_selected} — Demand (7-day avg) & Reorder Point")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# -----------------------
# HELP tab
# -----------------------
with tab_help:
    st.subheader("How to use (Weekly CSV)")
    st.markdown(
        """
**1) SALES CSV (weekly upload)**  
- Format: `Date, SKU, Quantity, Stock`  
- Daily totals per SKU.  
- Κάθε εβδομάδα ανεβάζεις νέο αρχείο → το app κάνει **append/replace** στο ιστορικό (unique Date+SKU).

**2) PRODUCTS CSV (optional)**  
- Format: `SKU, ProductName, Category, LeadTime, DaysToCover, ServiceLevel, MinStock`  
- Τα LeadTime/DaysToCover/ServiceLevel/MinStock είναι **overrides ανά SKU**.

**3) Tip**  
- Αν έχεις διαφορετικό stock rule (π.χ. min stock) βάλε `MinStock` στο PRODUCTS.
"""
    )
