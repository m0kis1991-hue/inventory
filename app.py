import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="InventoryAI",
    page_icon="📦",
    layout="wide"
)

st.title("📦 InventoryAI")
st.caption(
    "Inventory planning for small businesses — upload sales, track stock, forecast demand, "
    "and generate smarter reorder suggestions."
)

DB_PATH = "inventory_ai.sqlite"

# =========================================================
# DATABASE
# =========================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


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
            supplier TEXT,
            lead_time_override INTEGER,
            days_to_cover_override INTEGER,
            service_level_override INTEGER,
            min_stock_override REAL,
            unit_cost REAL,
            unit_price REAL,
            promo_min_qty REAL,
            promo_unit_cost REAL
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
        payload.append(
            (
                r["SKU"],
                r["ProductName"] if "ProductName" in df.columns else None,
                r["Category"] if "Category" in df.columns else None,
                r["Supplier"] if "Supplier" in df.columns else None,
                int(r["LeadTime"]) if "LeadTime" in df.columns and pd.notna(r["LeadTime"]) else None,
                int(r["DaysToCover"]) if "DaysToCover" in df.columns and pd.notna(r["DaysToCover"]) else None,
                int(r["ServiceLevel"]) if "ServiceLevel" in df.columns and pd.notna(r["ServiceLevel"]) else None,
                float(r["MinStock"]) if "MinStock" in df.columns and pd.notna(r["MinStock"]) else None,
                float(r["UnitCost"]) if "UnitCost" in df.columns and pd.notna(r["UnitCost"]) else None,
                float(r["UnitPrice"]) if "UnitPrice" in df.columns and pd.notna(r["UnitPrice"]) else None,
                float(r["PromoMinQty"]) if "PromoMinQty" in df.columns and pd.notna(r["PromoMinQty"]) else None,
                float(r["PromoUnitCost"]) if "PromoUnitCost" in df.columns and pd.notna(r["PromoUnitCost"]) else None,
            )
        )

    cur.executemany("""
        INSERT OR REPLACE INTO products
        (
            sku, product_name, category, supplier,
            lead_time_override, days_to_cover_override, service_level_override, min_stock_override,
            unit_cost, unit_price, promo_min_qty, promo_unit_cost
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            supplier AS Supplier,
            lead_time_override AS LeadTime,
            days_to_cover_override AS DaysToCover,
            service_level_override AS ServiceLevel,
            min_stock_override AS MinStock,
            unit_cost AS UnitCost,
            unit_price AS UnitPrice,
            promo_min_qty AS PromoMinQty,
            promo_unit_cost AS PromoUnitCost
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
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make incoming CSVs more forgiving:
    date/date sold -> Date
    sku/code/item code -> SKU
    quantity/qty/sales -> Quantity
    stock/inventory/on hand -> Stock
    """
    rename_map = {}
    for col in df.columns:
        c = str(col).strip().lower()

        if c in ["date", "day", "sales date", "sale date"]:
            rename_map[col] = "Date"
        elif c in ["sku", "code", "item code", "product code"]:
            rename_map[col] = "SKU"
        elif c in ["quantity", "qty", "sales", "units sold"]:
            rename_map[col] = "Quantity"
        elif c in ["stock", "inventory", "on hand", "stock level"]:
            rename_map[col] = "Stock"
        elif c in ["productname", "product name", "name"]:
            rename_map[col] = "ProductName"
        elif c in ["category", "group"]:
            rename_map[col] = "Category"
        elif c in ["supplier", "vendor"]:
            rename_map[col] = "Supplier"
        elif c in ["leadtime", "lead time"]:
            rename_map[col] = "LeadTime"
        elif c in ["daystocover", "days to cover"]:
            rename_map[col] = "DaysToCover"
        elif c in ["servicelevel", "service level"]:
            rename_map[col] = "ServiceLevel"
        elif c in ["minstock", "min stock"]:
            rename_map[col] = "MinStock"
        elif c in ["unitcost", "unit cost", "cost"]:
            rename_map[col] = "UnitCost"
        elif c in ["unitprice", "unit price", "price"]:
            rename_map[col] = "UnitPrice"
        elif c in ["promominqty", "promo min qty", "minimum promo qty"]:
            rename_map[col] = "PromoMinQty"
        elif c in ["promounitcost", "promo unit cost", "promo cost"]:
            rename_map[col] = "PromoUnitCost"

    return df.rename(columns=rename_map)


def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    required_cols = {"Date", "SKU", "Quantity", "Stock"}
    missing = required_cols - set(df.columns)

    if missing:
        st.error(f"❌ SALES CSV is missing: {', '.join(sorted(missing))}")
        st.info("Required columns: Date, SKU, Quantity, Stock")
        st.stop()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        st.error("❌ Date column contains invalid dates.")
        st.info("Example format: 2026-03-04")
        st.stop()

    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0.0)

    # If same Date+SKU appears multiple times in same upload, merge it
    df = (
        df.groupby(["Date", "SKU"], as_index=False)
        .agg({"Quantity": "sum", "Stock": "last"})
    )

    return df.sort_values(["SKU", "Date"]).reset_index(drop=True)


def validate_products(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    if "SKU" not in df.columns:
        st.error("❌ PRODUCTS CSV must contain SKU.")
        st.stop()

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()
    return df


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
        ["SKU001", "Paracetamol 500mg", "Pharmacy", "Supplier A", 7, 14, 95, 20, 1.50, 3.20, 100, 1.25],
        ["SKU002", "Vitamin C 1000mg", "Supplements", "Supplier B", 10, 20, 95, 10, 2.10, 4.90, "", ""],
    ], columns=[
        "SKU", "ProductName", "Category", "Supplier",
        "LeadTime", "DaysToCover", "ServiceLevel", "MinStock",
        "UnitCost", "UnitPrice", "PromoMinQty", "PromoUnitCost"
    ])
    return sample.to_csv(index=False).encode("utf-8")


@st.cache_data
def generate_demo_sales(num_skus=120, days=730):
    np.random.seed(42)
    skus = [f"SKU{i:04}" for i in range(1, num_skus + 1)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rows = []
    for sku in skus:
        base_lam = np.random.randint(2, 30)
        stock = np.random.randint(100, 600)

        for d in dates:
            # crude seasonality effect
            month = d.month
            season_boost = 1.0
            if month in [11, 12]:
                season_boost *= 1.20
            if month in [1, 2]:
                season_boost *= 1.05

            qty = np.random.poisson(lam=max(base_lam * season_boost, 0.1))
            replen = np.random.randint(0, 10)

            stock = stock - qty + replen
            if stock < 0:
                stock = np.random.randint(50, 250)

            rows.append([d, sku, float(qty), float(stock)])

    df = pd.DataFrame(rows, columns=["Date", "SKU", "Quantity", "Stock"])
    return df


@st.cache_data
def generate_demo_products(num_skus=120):
    categories = ["Pharmacy", "Supplements", "Hygiene", "Beverages", "Pet", "Grocery"]
    suppliers = ["Supplier A", "Supplier B", "Supplier C"]

    rows = []
    for i in range(1, num_skus + 1):
        sku = f"SKU{i:04}"
        category = categories[(i - 1) % len(categories)]
        supplier = suppliers[(i - 1) % len(suppliers)]

        unit_cost = round(np.random.uniform(1.0, 12.0), 2)
        unit_price = round(unit_cost * np.random.uniform(1.4, 2.2), 2)

        if i % 5 == 0:
            promo_min = 100
            promo_unit_cost = round(unit_cost * 0.90, 2)
        else:
            promo_min = None
            promo_unit_cost = None

        rows.append([
            sku,
            f"Demo Product {i}",
            category,
            supplier,
            None, None, None, None,
            unit_cost,
            unit_price,
            promo_min,
            promo_unit_cost
        ])

    return pd.DataFrame(rows, columns=[
        "SKU", "ProductName", "Category", "Supplier",
        "LeadTime", "DaysToCover", "ServiceLevel", "MinStock",
        "UnitCost", "UnitPrice", "PromoMinQty", "PromoUnitCost"
    ])


def current_event_multiplier(row, event_name, event_pct, keyword):
    """
    Manual event multipliers for current market situation.
    Example: Christmas, flu, tourism season, etc.
    """
    if event_name == "None" or event_pct == 0:
        return 1.0

    if keyword.strip() == "":
        return 1.0 + (event_pct / 100.0)

    haystack = " ".join([
        str(row.get("SKU", "")),
        str(row.get("ProductName", "")),
        str(row.get("Category", ""))
    ]).lower()

    if keyword.lower() in haystack:
        return 1.0 + (event_pct / 100.0)

    return 1.0


# =========================================================
# INIT DB
# =========================================================
init_db()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Inventory Strategy")

use_demo = st.sidebar.checkbox("Use demo sample data", value=False)

default_lead_time = st.sidebar.number_input("Default Lead Time (days)", min_value=1, value=7)
default_service_level = st.sidebar.slider("Default Service Level (%)", min_value=80, max_value=99, value=95)

default_days_to_cover = st.sidebar.number_input(
    "Target Inventory (days of stock)",
    min_value=1,
    value=14
)

default_min_stock = st.sidebar.number_input("Default Minimum Stock", min_value=0.0, value=0.0)

annual_holding_rate_pct = st.sidebar.number_input(
    "Annual Carrying Cost (%)",
    min_value=0.0,
    value=18.0,
    help="Estimated annual cost of holding extra stock."
)

st.sidebar.divider()
st.sidebar.subheader("📈 Market / Event Adjustment")

event_name = st.sidebar.selectbox(
    "Current event",
    ["None", "Christmas / Holidays", "Flu / Health Spike", "Tourism Season", "Heatwave", "Custom"]
)

event_pct = st.sidebar.number_input(
    "Event demand adjustment (%)",
    min_value=-50.0,
    max_value=200.0,
    value=0.0,
    help="Use positive values for expected increase in demand, negative for decrease."
)

event_keyword = st.sidebar.text_input(
    "Apply event only to keyword (optional)",
    value="",
    help="Example: pharmacy, cough, flu, toys, gift, sunscreen"
)

manual_market_growth_pct = st.sidebar.number_input(
    "Recent business growth / decline (%)",
    min_value=-80.0,
    max_value=200.0,
    value=0.0,
    help="Use this if you know your customer base has recently increased or decreased."
)

st.sidebar.divider()
st.sidebar.subheader("📤 Uploads")

sales_file = st.sidebar.file_uploader("Upload SALES CSV", type="csv")
products_file = st.sidebar.file_uploader("Upload PRODUCTS CSV (optional)", type="csv")

c1, c2 = st.sidebar.columns(2)
apply_upload = c1.button("✅ Apply Upload")
reset_all = c2.button("🧨 Reset Data")

if reset_all:
    st.sidebar.warning("This will delete ALL sales and product data.")
    confirm_reset = st.sidebar.checkbox("Confirm reset")
    if confirm_reset:
        reset_database()
        st.sidebar.success("✅ All data deleted. Refresh the page.")
        st.stop()

if apply_upload:
    if use_demo:
        n_sales = upsert_sales(generate_demo_sales())
        n_products = upsert_products(generate_demo_products())
        st.success(f"✅ Demo data loaded: {n_sales:,} sales rows and {n_products:,} products.")
    else:
        if sales_file is None:
            st.error("❌ Please upload SALES CSV first.")
            st.stop()

        sales_df = pd.read_csv(sales_file)
        sales_df = validate_sales(sales_df)
        n_sales = upsert_sales(sales_df)

        if products_file is not None:
            products_df = pd.read_csv(products_file)
            products_df = validate_products(products_df)
            n_products = upsert_products(products_df)
            st.success(f"✅ SALES uploaded: {n_sales:,} rows | PRODUCTS uploaded: {n_products:,} rows")
        else:
            st.success(f"✅ SALES uploaded: {n_sales:,} rows")

    st.info("If you do not see the new data immediately, refresh the page.")

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
    st.subheader("📤 Upload & Data History")

    st.markdown("""
Use this section to build your inventory intelligence over time.

You can upload:
- **SALES CSV** with sales that have already happened
- **PRODUCTS CSV** with your product catalog, supplier info, and stock rules

As you keep uploading new data, the app builds a stronger historical base and improves reorder decisions.
""")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download SALES template.csv",
        data=sales_template_csv(),
        file_name="sales_template.csv",
        mime="text/csv"
    )
    d2.download_button(
        "⬇️ Download PRODUCTS template.csv",
        data=products_template_csv(),
        file_name="products_template.csv",
        mime="text/csv"
    )

    st.divider()

    if sales.empty:
        st.warning("No sales data uploaded yet.")
    else:
        st.success(f"Stored sales history: {len(sales):,} rows")
        st.dataframe(sales.tail(20), use_container_width=True)

    if not products.empty:
        st.success(f"Stored products catalog: {len(products):,} products")

# =========================================================
# PRODUCTS TAB
# =========================================================
with tab_products:
    st.subheader("📦 Product Catalog")

    st.markdown("""
Your product catalog is optional, but it makes the app much more useful.

With a PRODUCTS file you can store:
- product names
- categories
- suppliers
- lead times
- stock targets
- supplier promo rules
""")

    if products.empty:
        st.info("No product catalog uploaded yet.")
    else:
        st.dataframe(products, use_container_width=True)

# =========================================================
# FORECAST TAB
# =========================================================
with tab_forecast:
    st.subheader("📊 Demand Forecast & Reorder Suggestions")

    if sales.empty:
        st.warning("No sales data found. Upload SALES CSV first.")
        st.stop()

    st.markdown("### Sales data preview")
    st.dataframe(sales.tail(50), use_container_width=True)

    # -----------------------------------------------------
    # VECTORISED FORECAST ENGINE FOR 1000+ SKUs
    # -----------------------------------------------------
    sales = sales.sort_values(["SKU", "Date"]).reset_index(drop=True)
    today = sales["Date"].max()
    current_month = int(today.month)

    # Base stats
    sku_stats = (
        sales.groupby("SKU", as_index=False)
        .agg(
            AvgAll=("Quantity", "mean"),
            StdAll=("Quantity", "std"),
            CurrentStock=("Stock", "last"),
            LastDate=("Date", "max"),
            DaysHistory=("Date", "count")
        )
    )

    sku_stats["StdAll"] = sku_stats["StdAll"].fillna(0.0)

    # Recent 28-day avg
    recent_start = today - pd.Timedelta(days=27)
    recent_df = sales[sales["Date"] >= recent_start]
    recent_avg = recent_df.groupby("SKU")["Quantity"].mean().rename("AvgRecent28")

    # Previous 28-day avg
    prev_start = today - pd.Timedelta(days=55)
    prev_end = today - pd.Timedelta(days=28)
    prev_df = sales[(sales["Date"] >= prev_start) & (sales["Date"] <= prev_end)]
    prev_avg = prev_df.groupby("SKU")["Quantity"].mean().rename("AvgPrev28")

    # Same period last year avg
    ly_start = recent_start - pd.Timedelta(days=365)
    ly_end = today - pd.Timedelta(days=365)
    ly_df = sales[(sales["Date"] >= ly_start) & (sales["Date"] <= ly_end)]
    ly_avg = ly_df.groupby("SKU")["Quantity"].mean().rename("AvgSamePeriodLastYear")

    # Current month seasonality
    sales["Month"] = sales["Date"].dt.month
    month_avg = sales.groupby(["SKU", "Month"])["Quantity"].mean().rename("MonthAvg").reset_index()
    overall_avg = sales.groupby("SKU")["Quantity"].mean().rename("OverallAvg")

    current_month_avg = month_avg[month_avg["Month"] == current_month][["SKU", "MonthAvg"]].rename(columns={"MonthAvg": "CurrentMonthAvg"})

    # Merge all stats
    sku_stats = sku_stats.merge(recent_avg, on="SKU", how="left")
    sku_stats = sku_stats.merge(prev_avg, on="SKU", how="left")
    sku_stats = sku_stats.merge(ly_avg, on="SKU", how="left")
    sku_stats = sku_stats.merge(overall_avg, on="SKU", how="left")
    sku_stats = sku_stats.merge(current_month_avg, on="SKU", how="left")

    sku_stats["AvgRecent28"] = sku_stats["AvgRecent28"].fillna(sku_stats["AvgAll"])
    sku_stats["AvgPrev28"] = sku_stats["AvgPrev28"].fillna(sku_stats["AvgAll"])
    sku_stats["AvgSamePeriodLastYear"] = sku_stats["AvgSamePeriodLastYear"].fillna(sku_stats["AvgAll"])
    sku_stats["OverallAvg"] = sku_stats["OverallAvg"].fillna(sku_stats["AvgAll"])
    sku_stats["CurrentMonthAvg"] = sku_stats["CurrentMonthAvg"].fillna(sku_stats["OverallAvg"])

    # Recent trend factor
    sku_stats["RecentTrendFactor"] = np.where(
        sku_stats["AvgPrev28"] > 0,
        sku_stats["AvgRecent28"] / sku_stats["AvgPrev28"],
        1.0
    )
    sku_stats["RecentTrendFactor"] = sku_stats["RecentTrendFactor"].clip(lower=0.70, upper=1.50)

    # Yearly seasonality factor
    sku_stats["SeasonalityFactor"] = np.where(
        sku_stats["OverallAvg"] > 0,
        sku_stats["CurrentMonthAvg"] / sku_stats["OverallAvg"],
        1.0
    )
    sku_stats["SeasonalityFactor"] = sku_stats["SeasonalityFactor"].clip(lower=0.70, upper=1.50)

    # Base blended forecast
    sku_stats["BaseForecastDaily"] = (
        0.45 * sku_stats["AvgRecent28"] +
        0.35 * sku_stats["AvgSamePeriodLastYear"] +
        0.20 * sku_stats["AvgAll"]
    )

    # Apply internal trend + seasonality
    sku_stats["ForecastDaily"] = (
        sku_stats["BaseForecastDaily"] *
        sku_stats["RecentTrendFactor"] *
        sku_stats["SeasonalityFactor"]
    )

    # Apply manual business growth / decline
    sku_stats["ForecastDaily"] = sku_stats["ForecastDaily"] * (1 + manual_market_growth_pct / 100.0)

    # Merge product catalog if available
    if not products.empty:
        sku_stats = sku_stats.merge(products, on="SKU", how="left")

    # Apply event multiplier row by row only after product fields merged
    sku_stats["EventMultiplier"] = sku_stats.apply(
        lambda r: current_event_multiplier(r, event_name, event_pct, event_keyword),
        axis=1
    )
    sku_stats["ForecastDaily"] = sku_stats["ForecastDaily"] * sku_stats["EventMultiplier"]

    # Final parameter values
    if "LeadTime" in sku_stats.columns:
        sku_stats["LeadTimeFinal"] = sku_stats["LeadTime"].fillna(default_lead_time)
    else:
        sku_stats["LeadTimeFinal"] = default_lead_time

    if "DaysToCover" in sku_stats.columns:
        sku_stats["DaysToCoverFinal"] = sku_stats["DaysToCover"].fillna(default_days_to_cover)
    else:
        sku_stats["DaysToCoverFinal"] = default_days_to_cover

    if "ServiceLevel" in sku_stats.columns:
        sku_stats["ServiceLevelFinal"] = sku_stats["ServiceLevel"].fillna(default_service_level)
    else:
        sku_stats["ServiceLevelFinal"] = default_service_level

    if "MinStock" in sku_stats.columns:
        sku_stats["MinStockFinal"] = sku_stats["MinStock"].fillna(default_min_stock)
    else:
        sku_stats["MinStockFinal"] = default_min_stock

    # Safety stock and reorder logic
    z_values = sku_stats["ServiceLevelFinal"].apply(lambda x: get_z(int(x)))
    sku_stats["SafetyStock"] = z_values * sku_stats["StdAll"].fillna(0.0) * np.sqrt(sku_stats["LeadTimeFinal"])
    sku_stats["ReorderPoint"] = (sku_stats["ForecastDaily"] * sku_stats["LeadTimeFinal"]) + sku_stats["SafetyStock"]
    sku_stats["TargetStock"] = (sku_stats["ForecastDaily"] * sku_stats["DaysToCoverFinal"]) + sku_stats["SafetyStock"]

    # Apply minimum stock floor
    sku_stats["ReorderPoint"] = np.maximum(sku_stats["ReorderPoint"], sku_stats["MinStockFinal"])
    sku_stats["TargetStock"] = np.maximum(sku_stats["TargetStock"], sku_stats["MinStockFinal"])

    sku_stats["OrderQty"] = (sku_stats["TargetStock"] - sku_stats["CurrentStock"]).clip(lower=0).round().astype(int)

    # Stockout days
    sku_stats["DaysOfCover"] = np.where(
        sku_stats["ForecastDaily"] > 0,
        (sku_stats["CurrentStock"] / sku_stats["ForecastDaily"]).round(1),
        np.nan
    )

    sku_stats["StockoutDays"] = np.where(
        sku_stats["ForecastDaily"] > 0,
        np.floor(sku_stats["CurrentStock"] / sku_stats["ForecastDaily"]),
        np.nan
    )

    sku_stats["Severity"] = (sku_stats["ReorderPoint"] - sku_stats["CurrentStock"]).clip(lower=0)
    sku_stats["Status"] = np.where(
        sku_stats["CurrentStock"] >= sku_stats["ReorderPoint"],
        "✅ OK",
        "⚠️ LOW STOCK"
    )

    # -----------------------------------------------------
    # PROMO / BULK BUY LOGIC
    # -----------------------------------------------------
    if "UnitCost" in sku_stats.columns and "PromoMinQty" in sku_stats.columns and "PromoUnitCost" in sku_stats.columns:
        sku_stats["UnitCost"] = pd.to_numeric(sku_stats["UnitCost"], errors="coerce")
        sku_stats["PromoMinQty"] = pd.to_numeric(sku_stats["PromoMinQty"], errors="coerce")
        sku_stats["PromoUnitCost"] = pd.to_numeric(sku_stats["PromoUnitCost"], errors="coerce")

        sku_stats["PromoSavingsPerUnit"] = (sku_stats["UnitCost"] - sku_stats["PromoUnitCost"]).clip(lower=0)
        sku_stats["ExtraUnitsForPromo"] = (sku_stats["PromoMinQty"] - sku_stats["OrderQty"]).clip(lower=0)

        daily_holding_rate = annual_holding_rate_pct / 100.0 / 365.0

        sku_stats["ExtraDaysHeld"] = np.where(
            sku_stats["ForecastDaily"] > 0,
            sku_stats["ExtraUnitsForPromo"] / sku_stats["ForecastDaily"],
            0
        )

        sku_stats["EstimatedHoldingCostExtra"] = (
            sku_stats["ExtraUnitsForPromo"] *
            sku_stats["UnitCost"].fillna(0) *
            daily_holding_rate *
            sku_stats["ExtraDaysHeld"]
        )

        sku_stats["EstimatedPromoSavings"] = (
            sku_stats["PromoMinQty"].fillna(0) *
            sku_stats["PromoSavingsPerUnit"].fillna(0)
        )

        sku_stats["PromoWorthIt"] = np.where(
            (
                sku_stats["PromoMinQty"].notna() &
                sku_stats["PromoUnitCost"].notna() &
                sku_stats["UnitCost"].notna() &
                (sku_stats["PromoUnitCost"] < sku_stats["UnitCost"]) &
                (sku_stats["PromoMinQty"] > sku_stats["OrderQty"]) &
                (sku_stats["EstimatedPromoSavings"] > sku_stats["EstimatedHoldingCostExtra"])
            ),
            "✅ YES",
            "—"
        )

        sku_stats["SuggestedOrderFinal"] = np.where(
            sku_stats["PromoWorthIt"] == "✅ YES",
            sku_stats["PromoMinQty"].fillna(sku_stats["OrderQty"]),
            sku_stats["OrderQty"]
        )
        sku_stats["SuggestedOrderFinal"] = sku_stats["SuggestedOrderFinal"].round().astype(int)

        sku_stats["NetPromoBenefit"] = (
            sku_stats["EstimatedPromoSavings"].fillna(0) -
            sku_stats["EstimatedHoldingCostExtra"].fillna(0)
        ).round(2)
    else:
        sku_stats["PromoWorthIt"] = "—"
        sku_stats["SuggestedOrderFinal"] = sku_stats["OrderQty"]
        sku_stats["NetPromoBenefit"] = 0.0

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------
    total_skus = len(sku_stats)
    low_count = int((sku_stats["Status"] == "⚠️ LOW STOCK").sum())
    ok_count = total_skus - low_count
    units_to_order = int(sku_stats["SuggestedOrderFinal"].sum())

    a, b, c, d = st.columns(4)
    a.metric("Products", f"{total_skus:,}")
    b.metric("LOW STOCK", low_count)
    c.metric("OK", ok_count)
    d.metric("Suggested Units to Order", f"{units_to_order:,}")

    st.markdown("### Top 5 most critical products")
    critical = sku_stats.sort_values(["Severity", "DaysOfCover"], ascending=[False, True]).head(5)

    critical_cols = [
        c for c in [
            "SKU", "ProductName", "Category",
            "CurrentStock", "ReorderPoint", "DaysOfCover",
            "StockoutDays", "SuggestedOrderFinal", "PromoWorthIt", "Status"
        ] if c in critical.columns
    ]
    st.dataframe(critical[critical_cols], use_container_width=True)

    st.divider()

    # Search + filter
    search = st.text_input("🔎 Search SKU / Product / Category / Supplier")
    filtered = sku_stats.copy()

    if search.strip():
        needle = search.lower()

        def contains(series):
            return series.astype(str).str.lower().str.contains(needle, na=False)

        mask = contains(filtered["SKU"])

        for optional_col in ["ProductName", "Category", "Supplier"]:
            if optional_col in filtered.columns:
                mask = mask | contains(filtered[optional_col])

        filtered = filtered[mask]

    show_only_orders = st.checkbox("Show only products needing reorder", value=True)
    if show_only_orders:
        filtered = filtered[filtered["SuggestedOrderFinal"] > 0]

    filtered = filtered.sort_values(["Severity", "SuggestedOrderFinal"], ascending=[False, False])

    st.markdown("### Reorder table")

    preferred_cols = [
        "SKU", "ProductName", "Category", "Supplier",
        "CurrentStock", "ForecastDaily", "DaysOfCover", "StockoutDays",
        "ReorderPoint", "TargetStock", "OrderQty", "SuggestedOrderFinal",
        "PromoWorthIt", "NetPromoBenefit",
        "LeadTimeFinal", "ServiceLevelFinal", "DaysToCoverFinal",
        "Status"
    ]
    display_cols = [c for c in preferred_cols if c in filtered.columns]

    display_df = filtered[display_cols].rename(columns={
        "CurrentStock": "Current Stock",
        "ForecastDaily": "Forecast Daily Demand",
        "DaysOfCover": "Days of Cover",
        "StockoutDays": "Stockout in (days)",
        "ReorderPoint": "Reorder Point",
        "TargetStock": "Target Stock",
        "OrderQty": "Base Order Qty",
        "SuggestedOrderFinal": "Suggested Order Qty",
        "PromoWorthIt": "Promo Opportunity",
        "NetPromoBenefit": "Est. Promo Net Benefit",
        "LeadTimeFinal": "Lead Time",
        "ServiceLevelFinal": "Service Level",
        "DaysToCoverFinal": "Days to Cover"
    })

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "⬇️ Download reorder_suggestions.csv",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="reorder_suggestions.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown("### SKU chart")
    sku_options = sorted(sku_stats["SKU"].tolist())
    selected_sku = st.selectbox("Choose SKU", options=sku_options)

    temp = sales[sales["SKU"] == selected_sku].sort_values("Date").copy()
    temp["Sales_7d_avg"] = temp["Quantity"].rolling(7, min_periods=1).mean()

    reorder_line = float(sku_stats.loc[sku_stats["SKU"] == selected_sku, "ReorderPoint"].iloc[0])
    forecast_line = float(sku_stats.loc[sku_stats["SKU"] == selected_sku, "ForecastDaily"].iloc[0])

    plt.figure(figsize=(12, 4))
    plt.plot(temp["Date"], temp["Sales_7d_avg"], label="7-day avg sales")
    plt.axhline(y=reorder_line, linestyle="--", label="Reorder Point")
    plt.axhline(y=forecast_line, linestyle=":", label="Forecast Daily Demand")
    plt.title(f"{selected_sku} — Demand, Forecast & Reorder Point")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# =========================================================
# HOW TO USE TAB
# =========================================================
with tab_help:
    st.subheader("📘 How to Use")

    st.markdown("""
## What this application does

InventoryAI helps a small business understand:

- what products it currently has
- how fast those products are selling
- which products risk going out of stock
- which products are overstocked
- how much should be ordered next
- whether a supplier promotion is financially worth it

The goal is to reduce **stockouts** without holding unnecessary stock.
""")

    st.markdown("""
## How the system makes reorder decisions

The application builds a demand estimate using:

1. **Recent sales trend**  
   It checks whether demand has recently increased or decreased.

2. **Historical seasonality**  
   It compares current season/month behavior with older patterns from previous periods.

3. **Same period last year**  
   If previous-year data exists, it uses that period to improve the estimate.

4. **Manual market adjustments**  
   You can apply market growth/decline and event adjustments such as:
   - Christmas / Holidays
   - Flu / Health spike
   - Tourism season
   - Custom events

5. **Inventory rules**  
   The app combines demand with:
   - supplier lead time
   - desired days of stock
   - service level
   - optional minimum stock
""")

    st.markdown("""
## What the user can control

The user can define:

- **Lead Time**  
  How many days a supplier needs to deliver

- **Target Inventory (days of stock)**  
  How long they want to have stock available

- **Service Level**  
  The safety margin against stockouts

- **Minimum Stock**  
  A minimum stock floor for important items

This means the user can decide if they want stock for:
- 7 days
- 14 days
- 30 days
- or any other period
""")

    st.markdown("""
## SALES CSV format

The required SALES file must contain:

- `Date`
- `SKU`
- `Quantity`
- `Stock`

### Example
Date,SKU,Quantity,Stock  
2026-03-01,SKU001,5,120  
2026-03-01,SKU002,3,80  
2026-03-02,SKU001,4,116  
""")

    st.markdown("""
## PRODUCTS CSV format (optional)

The optional PRODUCTS file can contain:

- `SKU`
- `ProductName`
- `Category`
- `Supplier`
- `LeadTime`
- `DaysToCover`
- `ServiceLevel`
- `MinStock`
- `UnitCost`
- `UnitPrice`
- `PromoMinQty`
- `PromoUnitCost`

This helps the app:
- show proper product names
- apply different stock rules by product
- calculate whether supplier promos are worth taking
""")

    st.markdown("""
## How supplier promotions are evaluated

If the supplier offers a lower price when you buy more stock, the app checks:

- the standard reorder quantity
- the minimum quantity needed for the promo
- the discount gained
- the estimated carrying cost of holding extra stock

If the expected promo savings are larger than the holding cost of the extra stock, the app flags a **Promo Opportunity**.
""")

    st.markdown("""
## Typical workflow

1. Upload your latest SALES CSV  
2. Optionally upload/update your PRODUCTS CSV  
3. Review:
   - low stock items
   - suggested order quantities
   - promo opportunities
   - estimated stockout days  
4. Download the reorder file and send it to your supplier
""")

    st.success("The user can read everything here inside the app. No separate PDF is required.")
