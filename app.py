import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------
st.set_page_config(page_title="Inventory Forecasting MVP", layout="wide")
st.title("📦 Inventory & Demand Forecasting MVP")

# --------------------
@st.cache_data
def generate_sample_data(num_skus: int = 10, days: int = 365) -> pd.DataFrame:
    """Generate sample sales + stock history for demo."""
    np.random.seed(42)
    skus = [f"SKU{i:03}" for i in range(1, num_skus + 1)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rows = []
    for sku in skus:
        # διαφορετικό "χαρακτήρα" ζήτησης ανά SKU
        base_lam = np.random.randint(5, 40)  # μέση ημερήσια ζήτηση
        stock = np.random.randint(200, 600)  # αρχικό απόθεμα

        for d in dates:
            qty = np.random.poisson(lam=base_lam)
            # μικρή τυχαία αναπλήρωση (σαν να έρχονται παραλαβές)
            replen = np.random.randint(0, 15)

            stock = stock - qty + replen
            if stock < 0:
                stock = np.random.randint(50, 250)

            rows.append([d.strftime("%Y-%m-%d"), sku, int(qty), int(stock)])

    return pd.DataFrame(rows, columns=["Date", "SKU", "Quantity", "Stock"])


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and coerce types safely."""
    required_cols = {"Date", "SKU", "Quantity", "Stock"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"❌ Το CSV δεν έχει τις απαιτούμενες στήλες: {', '.join(sorted(missing))}")
        st.info("✅ Απαιτούνται ακριβώς οι στήλες: Date, SKU, Quantity, Stock")
        st.stop()

    # Dates
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        st.error("❌ Η στήλη Date έχει τιμές που δεν αναγνωρίζονται ως ημερομηνία.")
        st.info("Παράδειγμα σωστής μορφής: 2026-03-04")
        st.stop()

    # Numeric columns
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0)

    # Clean SKU
    df["SKU"] = df["SKU"].astype(str).str.strip()

    # Sort by date per SKU (σημαντικό για current_stock)
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

    return df


def get_z(service_level: int) -> float:
    """Approx Z values for common service levels."""
    z_table = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.65, 99: 2.33}
    return z_table.get(service_level, 1.65)


# --------------------
st.sidebar.header("⚙️ Ρυθμίσεις")
use_sample = st.sidebar.checkbox("✅ Χρήση δείγματος (χωρίς CSV)", value=True)

if use_sample:
    uploaded_sales = None
    uploaded_products = None
else:
    uploaded_sales = st.sidebar.file_uploader("📁 Upload SALES CSV (Date,SKU,Quantity,Stock)", type="csv")
    uploaded_products = st.sidebar.file_uploader("📁 Upload PRODUCTS CSV (προαιρετικό)", type="csv")

products_df = None

if not use_sample and uploaded_sales is not None:
    df = pd.read_csv(uploaded_sales)
    df = validate_and_clean(df)
    st.success("✅ SALES CSV φορτώθηκε επιτυχώς!")

    if uploaded_products is not None:
        products_df = pd.read_csv(uploaded_products)
        # products csv πρέπει να έχει τουλάχιστον SKU για join
        if "SKU" not in products_df.columns:
            st.warning("Το PRODUCTS CSV δεν έχει στήλη 'SKU', άρα δεν μπορεί να γίνει ένωση.")
            products_df = None
        else:
            products_df["SKU"] = products_df["SKU"].astype(str).str.strip()
            st.success("✅ PRODUCTS CSV φορτώθηκε (θα εμφανίζονται ονόματα/κατηγορίες/τιμές).")

else:
    st.info("ℹ️ Χρησιμοποιούμε τυχαίο δείγμα (10 προϊόντα / 365 ημέρες) για δοκιμή.")
    df = generate_sample_data(num_skus=10, days=365)
    df = validate_and_clean(df)

# --------------------
st.subheader("📄 Δεδομένα (preview)")
st.dataframe(df.head(20), use_container_width=True)

# --------------------
st.header("📌 Forecasting & Reorder Suggestions")

lead_time = st.sidebar.number_input("Lead Time (μέρες)", min_value=1, value=7)
service_level = st.sidebar.slider("Service Level (%)", min_value=80, max_value=99, value=95)
days_to_cover = st.sidebar.number_input("Στόχος κάλυψης (ημέρες)", min_value=1, value=14)

Z = get_z(int(service_level))

# Προβλέψεις ανά SKU
results = []

for sku in df["SKU"].unique():
    temp = df[df["SKU"] == sku].copy()

    avg_demand = float(temp["Quantity"].mean())
    std_demand = float(temp["Quantity"].std(ddof=1)) if len(temp) > 1 else 0.0

    safety_stock = Z * std_demand * np.sqrt(lead_time)
    reorder_point = (avg_demand * lead_time) + safety_stock

    current_stock = float(temp["Stock"].iloc[-1])

    # Target stock για X ημέρες + safety
    target_stock = (avg_demand * days_to_cover) + safety_stock
    order_qty = max(0, int(round(target_stock - current_stock)))

    status = "✅ OK" if current_stock >= reorder_point else "⚠️ LOW STOCK"

    results.append(
        {
            "SKU": sku,
            "Avg Demand": round(avg_demand, 2),
            "Std Demand": round(std_demand, 2),
            "Safety Stock": round(safety_stock, 2),
            "Reorder Point": round(reorder_point, 2),
            "Target Stock": round(target_stock, 2),
            "Current Stock": int(round(current_stock)),
            "Order Qty": order_qty,
            "Status": status,
        }
    )

res_df = pd.DataFrame(results)
# ✅ Merge extra product info (optional)
if products_df is not None and "SKU" in products_df.columns:
    res_df = res_df.merge(products_df, on="SKU", how="left")

# Days of Cover & Severity
res_df["Days of Cover"] = res_df.apply(
    lambda r: round(r["Current Stock"] / r["Avg Demand"], 1) if r["Avg Demand"] > 0 else np.nan,
    axis=1,
)
res_df["Severity"] = (res_df["Reorder Point"] - res_df["Current Stock"]).clip(lower=0)

# --------------------
st.subheader("🧾 Executive Summary")
total_skus = len(res_df)
low_count = int(res_df["Status"].str.contains("LOW STOCK").sum())
ok_count = int(total_skus - low_count)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Σύνολο προϊόντων", total_skus)
c2.metric("⚠️ LOW STOCK", low_count)
c3.metric("✅ OK", ok_count)
c4.metric("Στόχος κάλυψης (ημέρες)", int(days_to_cover))

critical = res_df.sort_values(["Severity", "Days of Cover"], ascending=[False, True]).head(5)
st.markdown("**Top 5 πιο κρίσιμα προϊόντα**")
st.dataframe(
    critical[["SKU", "Current Stock", "Reorder Point", "Days of Cover", "Order Qty", "Status"]],
    use_container_width=True,
)

# --------------------
st.subheader("📊 Reorder Forecast (όλα τα προϊόντα)")

show_only_orders = st.checkbox("Δείξε μόνο προϊόντα που χρειάζονται παραγγελία (Order Qty > 0)", value=False)
table_df = res_df[res_df["Order Qty"] > 0].copy() if show_only_orders else res_df.copy()

# πιο χρήσιμη ταξινόμηση για επιχειρήσεις
table_df = table_df.sort_values(["Status", "Severity", "Order Qty"], ascending=[True, False, False])
# προτεραιότητα σε βασικές στήλες + product fields αν υπάρχουν
base_cols = ["SKU", "ProductName", "Category", "Current Stock", "Days of Cover", "Reorder Point", "Target Stock", "Order Qty", "Status"]
cols = [c for c in base_cols if c in table_df.columns] + [c for c in table_df.columns if c not in base_cols]

st.dataframe(table_df[cols], use_container_width=True)


# Download
csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Κατέβασε προτάσεις (CSV)",
    data=csv_bytes,
    file_name="reorder_suggestions.csv",
    mime="text/csv",
)

# --------------------
st.header("📈 Plots (7-day average sales)")

# Επιλογή SKU για να μην γεμίζει η σελίδα με 10 γραφήματα
sku_selected = st.selectbox("Διάλεξε προϊόν (SKU) για γράφημα:", options=sorted(df["SKU"].unique()))
temp = df[df["SKU"] == sku_selected].copy()

# rolling average πωλήσεων
temp["Sales_7d_avg"] = temp["Quantity"].rolling(7, min_periods=1).mean()

reorder_line = float(res_df.loc[res_df["SKU"] == sku_selected, "Reorder Point"].iloc[0])

plt.figure(figsize=(12, 4))
plt.plot(temp["Date"], temp["Sales_7d_avg"], label="7-day avg sales")
plt.axhline(y=reorder_line, linestyle="--", label="Reorder Point")
plt.title(f"{sku_selected} - Demand (7-day avg) & Reorder Point")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()