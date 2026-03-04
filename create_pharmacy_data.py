import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# Professional Pharmacy Demo Data Generator
# Creates:
# 1) data/pharmacy_sales_30days.csv  (Date, SKU, Quantity, Stock)  -> compatible with your Streamlit app
# 2) data/pharmacy_products.csv     (SKU, ProductName, Category, UnitCost, UnitPrice, LeadTimeDays)
# -----------------------------

np.random.seed(1)

# ✅ Output folder
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Product catalog (more realistic)
products = [
    {"SKU": "PH001_PARACETAMOL",   "ProductName": "Paracetamol 500mg (20 tabs)",      "Category": "Pain relief",     "UnitCost": 1.20, "UnitPrice": 2.50, "LeadTimeDays": 2, "DemandLevel": "HIGH"},
    {"SKU": "PH002_IBUPROFEN",     "ProductName": "Ibuprofen 400mg (20 tabs)",        "Category": "Pain relief",     "UnitCost": 1.50, "UnitPrice": 3.20, "LeadTimeDays": 2, "DemandLevel": "HIGH"},
    {"SKU": "PH003_VITAMIN_C",     "ProductName": "Vitamin C 1000mg (20 effervescent)","Category": "Vitamins",        "UnitCost": 2.10, "UnitPrice": 4.90, "LeadTimeDays": 3, "DemandLevel": "MED"},
    {"SKU": "PH004_COUGH_SYRUP",   "ProductName": "Cough Syrup 200ml",                 "Category": "Cold & Flu",      "UnitCost": 3.00, "UnitPrice": 6.90, "LeadTimeDays": 3, "DemandLevel": "MED"},
    {"SKU": "PH005_ANTIHISTAMINE", "ProductName": "Antihistamine (10 tabs)",           "Category": "Allergy",          "UnitCost": 2.80, "UnitPrice": 6.50, "LeadTimeDays": 4, "DemandLevel": "MED"},
    {"SKU": "PH006_OMEGA3",        "ProductName": "Omega-3 (60 capsules)",             "Category": "Supplements",      "UnitCost": 4.50, "UnitPrice": 9.90, "LeadTimeDays": 5, "DemandLevel": "LOW"},
    {"SKU": "PH007_MAGNESIUM",     "ProductName": "Magnesium 300mg (60 tabs)",         "Category": "Supplements",      "UnitCost": 3.80, "UnitPrice": 8.90, "LeadTimeDays": 5, "DemandLevel": "MED"},
    {"SKU": "PH008_PROBIOTIC",     "ProductName": "Probiotic (20 capsules)",           "Category": "Digestive health", "UnitCost": 5.50, "UnitPrice": 12.90,"LeadTimeDays": 6, "DemandLevel": "LOW"},
    {"SKU": "PH009_SUNSCREEN",     "ProductName": "Sunscreen SPF50 50ml",              "Category": "Dermocosmetics",   "UnitCost": 6.00, "UnitPrice": 14.90,"LeadTimeDays": 6, "DemandLevel": "LOW"},
    {"SKU": "PH010_HANDGEL",       "ProductName": "Hand Sanitizer Gel 100ml",          "Category": "Hygiene",          "UnitCost": 0.90, "UnitPrice": 2.20, "LeadTimeDays": 3, "DemandLevel": "MED"},
]

products_df = pd.DataFrame(products)
products_df.to_csv(os.path.join(OUTPUT_DIR, "pharmacy_products.csv"), index=False)

# ✅ Dates: last 30 days (including today)
days = 30
end_date = datetime.today().date()
start_date = end_date - timedelta(days=days - 1)
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# ✅ Demand profiles (Poisson λ)
# High movers sell more, low movers sell less
DEMAND_LAM = {
    "HIGH": (6, 14),  # range -> we sample per SKU so each SKU differs slightly
    "MED":  (3, 7),
    "LOW":  (0.5, 2.5),
}

# ✅ Seasonality: weekends slightly higher traffic (Sat/Sun)
def weekend_multiplier(d: pd.Timestamp) -> float:
    return 1.15 if d.weekday() >= 5 else 1.0

sales_rows = []

for p in products:
    sku = p["SKU"]
    level = p["DemandLevel"]

    lam_low, lam_high = DEMAND_LAM[level]
    base_lam = np.random.uniform(lam_low, lam_high)

    # initial stock depends on demand level
    if level == "HIGH":
        stock = np.random.randint(120, 260)
    elif level == "MED":
        stock = np.random.randint(90, 200)
    else:
        stock = np.random.randint(60, 140)

    # restock policy: if stock goes below threshold, a delivery arrives after lead time
    lead_time = int(p["LeadTimeDays"])
    reorder_threshold = int(max(15, round(base_lam * 4)))  # ~4 days of demand
    scheduled_delivery = {}  # date -> quantity

    for d in dates:
        # delivery arrives today?
        if d.date() in scheduled_delivery:
            stock += scheduled_delivery[d.date()]

        lam_today = base_lam * weekend_multiplier(d)
        qty = np.random.poisson(lam=max(lam_today, 0.1))

        # cannot sell more than stock on hand
        qty = int(min(qty, stock))
        stock = stock - qty

        # schedule delivery if below threshold and no delivery already scheduled
        # delivery size: enough for days + buffer
        if stock <= reorder_threshold:
            delivery_date = (d + pd.Timedelta(days=lead_time)).date()
            if delivery_date not in scheduled_delivery:
                delivery_qty = int(round(base_lam * 12 + np.random.randint(10, 40)))  # ~12 days cover
                scheduled_delivery[delivery_date] = delivery_qty

        sales_rows.append([d.strftime("%Y-%m-%d"), sku, qty, stock])

sales_df = pd.DataFrame(sales_rows, columns=["Date", "SKU", "Quantity", "Stock"])
sales_path = os.path.join(OUTPUT_DIR, "pharmacy_sales_30days.csv")
sales_df.to_csv(sales_path, index=False)

print("✅ Έτοιμο demo dataset φαρμακείου!")
print(f"1) Sales file:    {sales_path}")
print(f"2) Products file: {os.path.join(OUTPUT_DIR, 'pharmacy_products.csv')}")
print("Tip: Ανέβασε το 'pharmacy_sales_30days.csv' στο Streamlit app σου.")