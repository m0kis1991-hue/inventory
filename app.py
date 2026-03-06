import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="InventoryAI",
    page_icon="📦",
    layout="wide"
)

DB_PATH = "inventory_ai.sqlite"

# =========================================================
# DATABASE
# =========================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            date TEXT NOT NULL,
            sku TEXT NOT NULL,
            quantity REAL NOT NULL,
            stock REAL NOT NULL,
            PRIMARY KEY (date, sku)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            sku TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            lead_time_override INTEGER,
            days_to_cover_override INTEGER,
            service_level_override INTEGER,
            min_stock_override REAL
        )
    """)

    conn.commit()
    conn.close()


def reset_database():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sales")
    cur.execute("DELETE FROM products")
    conn.commit()
    conn.close()


def upsert_sales(df: pd.DataFrame):
    conn = get_conn()
    cur = conn.cursor()

    rows = df.copy()
    rows["Date"] = pd.to_datetime(rows["Date"]).dt.strftime("%Y-%m-%d")
    rows["SKU"] = rows["SKU"].astype(str).str.strip()

    payload = list(rows[["Date", "SKU", "Quantity", "Stock"]].itertuples(index=False, name=None))

    cur.executemany("""
        INSERT OR REPLACE INTO sales (date, sku, quantity, stock)
        VALUES (?, ?, ?, ?)
    """, payload)

    conn.commit()
    conn.close()
    return len(payload)


def upsert_products(df: pd.DataFrame):
    conn = get_conn()
    cur = conn.cursor()

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    payload = []
    for _, r in df.iterrows():
        payload.append((
            r["SKU"],
            r["ProductName"] if "ProductName" in df.columns else None,
            r["Category"] if "Category" in df.columns else None,
            int(r["LeadTime"]) if "LeadTime" in df.columns and pd.notna(r["LeadTime"]) else None,
            int(r["DaysToCover"]) if "DaysToCover" in df.columns and pd.notna(r["DaysToCover"]) else None,
            int(r["ServiceLevel"]) if "ServiceLevel" in df.columns and pd.notna(r["ServiceLevel"]) else None,
            float(r["MinStock"]) if "MinStock" in df.columns and pd.notna(r["MinStock"]) else None
        ))

    cur.executemany("""
        INSERT OR REPLACE INTO products
        (sku, product_name, category, lead_time_override, days_to_cover_override, service_level_override, min_stock_override)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, payload)

    conn.commit()
    conn.close()
    return len(payload)


def load_sales():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            date AS Date,
            sku AS SKU,
            quantity AS Quantity,
            stock AS Stock
        FROM sales
    """, conn)
    conn.close()

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0.0)

    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)
    return df


def load_products():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            sku AS SKU,
            product_name AS ProductName,
            category AS Category,
            lead_time_override AS LeadTime,
            days_to_cover_override AS DaysToCover,
            service_level_override AS ServiceLevel,
            min_stock_override AS MinStock
        FROM products
    """, conn)
    conn.close()

    if df.empty:
        return df

    df["SKU"] = df["SKU"].astype(str).str.strip()
    return df


# =========================================================
# HELPERS
# =========================================================
def get_z(service_level: int) -> float:
    z_table = {
        80: 0.84,
        85: 1.04,
        90: 1.28,
        95: 1.65,
        97: 1.88,
        98: 2.05,
        99: 2.33
    }
    return z_table.get(int(service_level), 1.65)


def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Date", "SKU", "Quantity", "Stock"}
    missing = required_cols - set(df.columns)

    if missing:
        st.error(f"❌ SALES CSV λείπουν στήλες: {', '.join(sorted(missing))}")
        st.info("Απαιτούνται ακριβώς: Date, SKU, Quantity, Stock")
        st.stop()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        st.error("❌ Η στήλη Date έχει λάθος μορφή ημερομηνίας.")
        st.info("Παράδειγμα σωστής μορφής: 2026-03-04")
        st.stop()

    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0.0)

    # Daily totals mode: merge duplicates inside same upload
    df = (
        df.groupby(["Date", "SKU"], as_index=False)
        .agg({
            "Quantity": "sum",
            "Stock": "last"
        })
    )

    return df.sort_values(["SKU", "Date"]).reset_index(drop=True)


def validate_products(df: pd.DataFrame) -> pd.DataFrame:
    if "SKU" not in df.columns:
        st.error("❌ PRODUCTS CSV πρέπει να έχει τουλάχιστον στήλη SKU.")
        st.stop()

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()
    return df


def sales_template_csv():
    sample = pd.DataFrame([
        ["2026-03-01", "SKU001", 5, 120],
        ["2026-03-01", "SKU002", 3, 80],
        ["2026-03-02", "SKU001", 4, 116],
        ["2026-03-02", "SKU002", 2, 78],
    ], columns=["Date", "SKU", "Quantity", "Stock"])
    return sample.to_csv(index=False).encode("utf-8")


def products_template_csv():
    sample = pd.DataFrame([
        ["SKU001", "Paracetamol 500mg", "Pharmacy", 7, 14, 95, 20],
        ["SKU002", "Vitamin C 1000mg", "Supplements", 10, 20, 95, 10],
    ], columns=["SKU", "ProductName", "Category", "LeadTime", "DaysToCover", "ServiceLevel", "MinStock"])
    return sample.to_csv(index=False).encode("utf-8")


@st.cache_data
def generate_demo_sales(num_skus=50, days=365):
    np.random.seed(42)
    skus = [f"SKU{i:04}" for i in range(1, num_skus + 1)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rows = []
    for sku in skus:
        base_lam = np.random.randint(2, 30)
        stock = np.random.randint(100, 600)

        for d in dates:
            qty = np.random.poisson(lam=base_lam)
            replen = np.random.randint(0, 10)

            stock = stock - qty + replen
            if stock < 0:
                stock = np.random.randint(50, 250)

            rows.append([d, sku, float(qty), float(stock)])

    df = pd.DataFrame(rows, columns=["Date", "SKU", "Quantity", "Stock"])
    return df


@st.cache_data
def generate_demo_products(num_skus=50):
    return pd.DataFrame({
        "SKU": [f"SKU{i:04}" for i in range(1, num_skus + 1)],
        "ProductName": [f"Demo Product {i}" for i in range(1, num_skus + 1)],
        "Category": ["Demo"] * num_skus,
        "LeadTime": [None] * num_skus,
        "DaysToCover": [None] * num_skus,
        "ServiceLevel": [None] * num_skus,
        "MinStock": [None] * num_skus
    })


# =========================================================
# INIT
# =========================================================
init_db()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Global Defaults")

use_demo = st.sidebar.checkbox("Use demo sample data", value=False)

default_lead_time = st.sidebar.number_input("Default Lead Time (days)", min_value=1, value=7)
default_service_level = st.sidebar.slider("Default Service Level (%)", min_value=80, max_value=99, value=95)
default_days_to_cover = st.sidebar.number_input("Default Days to Cover", min_value=1, value=14)
default_min_stock = st.sidebar.number_input("Default Min Stock", min_value=0.0, value=0.0)

st.sidebar.divider()
st.sidebar.subheader("📁 Uploads")

sales_file = st.sidebar.file_uploader("Upload SALES CSV (weekly)", type="csv")
products_file = st.sidebar.file_uploader("Upload PRODUCTS CSV (optional)", type="csv")

col1, col2 = st.sidebar.columns(2)
apply_upload = col1.button("✅ Apply Upload")
reset_all = col2.button("🧨 Reset Data")

if reset_all:
    st.sidebar.warning("Αυτό θα σβήσει ΟΛΑ τα δεδομένα.")
    confirm_reset = st.sidebar.checkbox("Confirm reset")
    if confirm_reset:
        reset_database()
        st.sidebar.success("✅ Όλα τα δεδομένα σβήστηκαν. Κάνε refresh.")
        st.stop()

if apply_upload:
    if use_demo:
        n1 = upsert_sales(generate_demo_sales(num_skus=50, days=365))
        n2 = upsert_products(generate_demo_products(num_skus=50))
        st.success(f"✅ Demo data loaded: {n1} sales rows, {n2} products")
    else:
        if sales_file is None:
            st.error("❌ Πρέπει να ανεβάσεις SALES CSV.")
            st.stop()

        sales_df = pd.read_csv(sales_file)
        sales_df = validate_sales(sales_df)
        n_sales = upsert_sales(sales_df)

        if products_file is not None:
            products_df = pd.read_csv(products_file)
            products_df = validate_products(products_df)
            n_products = upsert_products(products_df)
            st.success(f"✅ SALES uploaded: {n_sales} rows | PRODUCTS uploaded: {n_products} rows")
        else:
            st.success(f"✅ SALES uploaded: {n_sales} rows")

    st.info("Αν δεν δεις αμέσως τις αλλαγές, κάνε refresh.")

# =========================================================
# LOAD DATA
# =========================================================
sales = load_sales()
products = load_products()

# =========================================================
# TABS
# =========================================================
tab_uploads, tab_products, tab_forecast, tab_help = st.tabs(
    ["📤 Uploads", "📦 Products", "📊 Forecast", "📘 How to Use"]
)

# =========================================================
# UPLOADS TAB
# =========================================================
with tab_uploads:
    st.subheader("📤 Weekly Upload Workflow")

    st.markdown("""
Αυτό το app είναι φτιαγμένο για **weekly upload**.

### Τι κάνεις κάθε εβδομάδα
1. Εξάγεις από POS / ERP / Excel το **SALES CSV**
2. Το ανεβάζεις από το sidebar
3. Πατάς **Apply Upload**
4. Πηγαίνεις στο **Forecast**
5. Κατεβάζεις το **reorder_suggestions.csv**
""")

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Download SALES template.csv",
        data=sales_template_csv(),
        file_name="sales_template.csv",
        mime="text/csv"
    )
    c2.download_button(
        "⬇️ Download PRODUCTS template.csv",
        data=products_template_csv(),
        file_name="products_template.csv",
        mime="text/csv"
    )

    st.divider()

    if sales.empty:
        st.warning("Δεν υπάρχουν ακόμα SALES δεδομένα.")
    else:
        st.success(f"Υπάρχουν αποθηκευμένες {len(sales):,} γραμμές SALES.")
        st.dataframe(sales.tail(20), use_container_width=True)

# =========================================================
# PRODUCTS TAB
# =========================================================
with tab_products:
    st.subheader("📦 Product Catalog")

    if products.empty:
        st.info("Δεν υπάρχει PRODUCTS catalog ακόμα. Είναι προαιρετικό.")
    else:
        st.dataframe(products, use_container_width=True)

    st.divider()
    st.subheader("➕ Quick Add / Update Product")

    p1, p2, p3 = st.columns(3)
    sku_input = p1.text_input("SKU")
    name_input = p2.text_input("ProductName")
    category_input = p3.text_input("Category")

    p4, p5, p6, p7 = st.columns(4)
    lead_input = p4.number_input("LeadTime", min_value=0, value=0)
    cover_input = p5.number_input("DaysToCover", min_value=0, value=0)
    service_input = p6.number_input("ServiceLevel", min_value=0, max_value=99, value=0)
    minstock_input = p7.number_input("MinStock", min_value=0.0, value=0.0)

    if st.button("💾 Save Product"):
        if not sku_input.strip():
            st.error("SKU είναι υποχρεωτικό.")
        else:
            one = pd.DataFrame([{
                "SKU": sku_input.strip(),
                "ProductName": name_input.strip() if name_input else None,
                "Category": category_input.strip() if category_input else None,
                "LeadTime": lead_input if lead_input > 0 else None,
                "DaysToCover": cover_input if cover_input > 0 else None,
                "ServiceLevel": service_input if service_input > 0 else None,
                "MinStock": minstock_input if minstock_input > 0 else None
            }])
            upsert_products(one)
            st.success("✅ Product saved. Κάνε refresh για να το δεις.")

# =========================================================
# FORECAST TAB
# =========================================================
with tab_forecast:
    st.subheader("📊 Forecast & Reorder Suggestions")

    if sales.empty:
        st.warning("Δεν υπάρχουν SALES δεδομένα. Κάνε upload πρώτα.")
        st.stop()

    st.markdown("### Data Preview")
    st.dataframe(sales.tail(50), use_container_width=True)

    # -----------------------------------------------------
    # VECTORISED STATS FOR 1000+ SKUs
    # -----------------------------------------------------
    sku_stats = (
        sales.sort_values(["SKU", "Date"])
        .groupby("SKU", as_index=False)
        .agg(
            AvgDemand=("Quantity", "mean"),
            StdDemand=("Quantity", "std"),
            CurrentStock=("Stock", "last"),
            LastDate=("Date", "max")
        )
    )

    sku_stats["StdDemand"] = sku_stats["StdDemand"].fillna(0.0)

    # Merge product catalog if exists
    if not products.empty:
        sku_stats = sku_stats.merge(products, on="SKU", how="left")

    # Apply overrides or defaults
    sku_stats["LeadTimeFinal"] = sku_stats["LeadTime"].fillna(default_lead_time) if "LeadTime" in sku_stats.columns else default_lead_time
    sku_stats["DaysToCoverFinal"] = sku_stats["DaysToCover"].fillna(default_days_to_cover) if "DaysToCover" in sku_stats.columns else default_days_to_cover
    sku_stats["ServiceLevelFinal"] = sku_stats["ServiceLevel"].fillna(default_service_level) if "ServiceLevel" in sku_stats.columns else default_service_level
    sku_stats["MinStockFinal"] = sku_stats["MinStock"].fillna(default_min_stock) if "MinStock" in sku_stats.columns else default_min_stock

    # Safety stock, reorder point, target stock
    z_values = sku_stats["ServiceLevelFinal"].apply(lambda x: get_z(int(x)))
    sku_stats["SafetyStock"] = z_values * sku_stats["StdDemand"] * np.sqrt(sku_stats["LeadTimeFinal"])
    sku_stats["ReorderPoint"] = (sku_stats["AvgDemand"] * sku_stats["LeadTimeFinal"]) + sku_stats["SafetyStock"]
    sku_stats["TargetStock"] = (sku_stats["AvgDemand"] * sku_stats["DaysToCoverFinal"]) + sku_stats["SafetyStock"]

    # Apply min stock floor
    sku_stats["ReorderPoint"] = np.maximum(sku_stats["ReorderPoint"], sku_stats["MinStockFinal"])
    sku_stats["TargetStock"] = np.maximum(sku_stats["TargetStock"], sku_stats["MinStockFinal"])

    sku_stats["OrderQty"] = (sku_stats["TargetStock"] - sku_stats["CurrentStock"]).clip(lower=0).round().astype(int)
    sku_stats["DaysOfCover"] = np.where(
        sku_stats["AvgDemand"] > 0,
        (sku_stats["CurrentStock"] / sku_stats["AvgDemand"]).round(1),
        np.nan
    )
    sku_stats["StockoutDays"] = np.where(
        sku_stats["AvgDemand"] > 0,
        np.floor(sku_stats["CurrentStock"] / sku_stats["AvgDemand"]).astype(int),
        np.nan
    )
    sku_stats["Severity"] = (sku_stats["ReorderPoint"] - sku_stats["CurrentStock"]).clip(lower=0)
    sku_stats["Status"] = np.where(
        sku_stats["CurrentStock"] >= sku_stats["ReorderPoint"],
        "✅ OK",
        "⚠️ LOW STOCK"
    )

    # Summary
    total_skus = len(sku_stats)
    low_count = int((sku_stats["Status"] == "⚠️ LOW STOCK").sum())
    ok_count = total_skus - low_count
    units_to_order = int(sku_stats["OrderQty"].sum())

    a, b, c, d = st.columns(4)
    a.metric("Products", f"{total_skus:,}")
    b.metric("LOW STOCK", low_count)
    c.metric("OK", ok_count)
    d.metric("Units to Order", f"{units_to_order:,}")

    st.markdown("### Top 5 Most Critical")
    critical = sku_stats.sort_values(["Severity", "DaysOfCover"], ascending=[False, True]).head(5)

    critical_cols = [
        c for c in [
            "SKU", "ProductName", "Category",
            "CurrentStock", "ReorderPoint",
            "DaysOfCover", "StockoutDays",
            "OrderQty", "Status"
        ] if c in critical.columns
    ]
    st.dataframe(critical[critical_cols], use_container_width=True)

    st.divider()

    # Search
    search = st.text_input("🔎 Search SKU / Product / Category")
    filtered = sku_stats.copy()

    if search.strip():
        search_lower = search.lower()

        def contains_series(series):
            return series.astype(str).str.lower().str.contains(search_lower, na=False)

        mask = contains_series(filtered["SKU"])
        if "ProductName" in filtered.columns:
            mask = mask | contains_series(filtered["ProductName"])
        if "Category" in filtered.columns:
            mask = mask | contains_series(filtered["Category"])

        filtered = filtered[mask]

    show_only_order = st.checkbox("Show only products needing order", value=True)
    if show_only_order:
        filtered = filtered[filtered["OrderQty"] > 0]

    filtered = filtered.sort_values(["Severity", "OrderQty"], ascending=[False, False])

    st.markdown("### Reorder Table")

    table_cols = [
        c for c in [
            "SKU", "ProductName", "Category",
            "CurrentStock", "AvgDemand", "DaysOfCover", "StockoutDays",
            "ReorderPoint", "TargetStock", "OrderQty",
            "LeadTimeFinal", "ServiceLevelFinal", "DaysToCoverFinal",
            "Status"
        ] if c in filtered.columns
    ]

    display_df = filtered[table_cols].rename(columns={
        "LeadTimeFinal": "LeadTime",
        "ServiceLevelFinal": "ServiceLevel",
        "DaysToCoverFinal": "DaysToCover",
        "AvgDemand": "Avg Demand",
        "CurrentStock": "Current Stock",
        "DaysOfCover": "Days of Cover",
        "StockoutDays": "Stockout in (days)",
        "ReorderPoint": "Reorder Point",
        "TargetStock": "Target Stock",
        "OrderQty": "Order Qty"
    })

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "⬇️ Download reorder_suggestions.csv",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="reorder_suggestions.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown("### SKU Chart")
    sku_options = sorted(sku_stats["SKU"].tolist())
    selected_sku = st.selectbox("Choose SKU", options=sku_options)

    temp = sales[sales["SKU"] == selected_sku].sort_values("Date").copy()
    temp["Sales_7d_avg"] = temp["Quantity"].rolling(7, min_periods=1).mean()

    reorder_line = float(sku_stats.loc[sku_stats["SKU"] == selected_sku, "ReorderPoint"].iloc[0])

    plt.figure(figsize=(12, 4))
    plt.plot(temp["Date"], temp["Sales_7d_avg"], label="7-day avg sales")
    plt.axhline(y=reorder_line, linestyle="--", label="Reorder Point")
    plt.title(f"{selected_sku} - Demand & Reorder Point")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# =========================================================
# HELP TAB
# =========================================================
with tab_help:
    st.subheader("📘 How to Use InventoryAI")

    st.markdown("""
## 1. What this app does
Το InventoryAI σε βοηθά να:

- ανεβάζεις κάθε εβδομάδα τα στοιχεία πωλήσεων σου
- βλέπεις ποια προϊόντα κινδυνεύουν να τελειώσουν
- βλέπεις πόσα τεμάχια πρέπει να παραγγείλεις
- κατεβάζεις αρχείο παραγγελίας για τον προμηθευτή
""")

    st.markdown("""
## 2. SALES CSV format
Το βασικό αρχείο που χρειάζεται το app είναι το **SALES CSV**.

Πρέπει να έχει ακριβώς αυτές τις στήλες:

- `Date`
- `SKU`
- `Quantity`
- `Stock`

### Παράδειγμα
```csv
Date,SKU,Quantity,Stock
2026-03-01,SKU001,5,120
2026-03-01,SKU002,3,80
2026-03-02,SKU001,4,116
