import os
import sqlite3
import hashlib
from datetime import datetime, timezone

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

st.title("📦 InventoryAI")
st.caption(
    "Inventory planning for small businesses — upload sales, track stock, "
    "forecast demand, and generate smarter reorder suggestions."
)

DB_PATH = "inventory_ai.sqlite"
DEFAULT_TRIAL_DAYS = 14

# =========================================================
# SECURITY
# =========================================================
def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000
    )
    return salt.hex(), hashed.hex()


def verify_password(password: str, salt_hex: str, password_hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000
    )
    return hashed.hex() == password_hash_hex


# =========================================================
# DATABASE
# =========================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def column_exists(cur, table_name: str, col_name: str) -> bool:
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    return col_name in cols


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # USERS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            store_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)

    # Migrations for users
    user_new_columns = {
        "email": "TEXT",
        "plan_name": "TEXT DEFAULT 'trial'",
        "subscription_status": "TEXT DEFAULT 'trial'",
        "trial_start": "TEXT",
        "trial_days": f"INTEGER DEFAULT {DEFAULT_TRIAL_DAYS}",
    }
    for col_name, col_type in user_new_columns.items():
        if not column_exists(cur, "users", col_name):
            cur.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")

    # SALES
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            date TEXT NOT NULL,
            sku TEXT NOT NULL,
            quantity REAL NOT NULL,
            stock REAL NOT NULL
        )
    """)

    if not column_exists(cur, "sales", "user_id"):
        cur.execute("ALTER TABLE sales ADD COLUMN user_id INTEGER DEFAULT 1")

    # PRODUCTS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            sku TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT
        )
    """)

    product_new_columns = {
        "user_id": "INTEGER DEFAULT 1",
        "supplier": "TEXT",
        "lead_time_override": "INTEGER",
        "days_to_cover_override": "INTEGER",
        "service_level_override": "INTEGER",
        "min_stock_override": "REAL",
        "unit_cost": "REAL",
        "unit_price": "REAL",
        "promo_min_qty": "REAL",
        "promo_unit_cost": "REAL",
    }
    for col_name, col_type in product_new_columns.items():
        if not column_exists(cur, "products", col_name):
            cur.execute(f"ALTER TABLE products ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def get_user_count() -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]
    conn.close()
    return count


def create_user(username: str, password: str, store_name: str, email: str, role: str = "user") -> tuple[bool, str]:
    conn = get_conn()
    cur = conn.cursor()

    try:
        salt_hex, hash_hex = hash_password(password)
        now_iso = datetime.now(timezone.utc).isoformat()

        cur.execute("""
            INSERT INTO users (
                username, password_salt, password_hash, store_name, role, created_at,
                email, plan_name, subscription_status, trial_start, trial_days
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username.strip(),
            salt_hex,
            hash_hex,
            store_name.strip(),
            role,
            now_iso,
            email.strip(),
            "trial" if role == "user" else "admin",
            "trial" if role == "user" else "active",
            now_iso if role == "user" else None,
            DEFAULT_TRIAL_DAYS if role == "user" else 0
        ))
        conn.commit()
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id, username, password_salt, password_hash, store_name, role,
            email, plan_name, subscription_status, trial_start, trial_days
        FROM users
        WHERE username = ?
    """, (username.strip(),))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    (
        user_id, db_username, salt_hex, hash_hex, store_name, role,
        email, plan_name, subscription_status, trial_start, trial_days
    ) = row

    if not verify_password(password, salt_hex, hash_hex):
        return None

    return {
        "id": user_id,
        "username": db_username,
        "store_name": store_name,
        "role": role,
        "email": email,
        "plan_name": plan_name,
        "subscription_status": subscription_status,
        "trial_start": trial_start,
        "trial_days": trial_days
    }


def list_users():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            id, username, email, store_name, role,
            plan_name, subscription_status, trial_start, trial_days, created_at
        FROM users
        ORDER BY id
    """, conn)
    conn.close()
    return df


def update_user_plan(user_id: int, plan_name: str, subscription_status: str, trial_days: int | None = None):
    conn = get_conn()
    cur = conn.cursor()

    if trial_days is None:
        cur.execute("""
            UPDATE users
            SET plan_name = ?, subscription_status = ?
            WHERE id = ?
        """, (plan_name, subscription_status, user_id))
    else:
        cur.execute("""
            UPDATE users
            SET plan_name = ?, subscription_status = ?, trial_days = ?
            WHERE id = ?
        """, (plan_name, subscription_status, trial_days, user_id))

    conn.commit()
    conn.close()


def delete_user(user_id: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM sales WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM products WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))

    conn.commit()
    conn.close()


def reset_user_data(user_id: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM sales WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM products WHERE user_id = ?", (user_id,))

    conn.commit()
    conn.close()


def upsert_sales(user_id: int, df: pd.DataFrame) -> int:
    conn = get_conn()
    cur = conn.cursor()

    rows = df.copy()
    rows["Date"] = pd.to_datetime(rows["Date"]).dt.strftime("%Y-%m-%d")
    rows["SKU"] = rows["SKU"].astype(str).str.strip()

    payload = [
        (user_id, row.Date, row.SKU, float(row.Quantity), float(row.Stock))
        for row in rows.itertuples(index=False)
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO sales (user_id, date, sku, quantity, stock)
        VALUES (?, ?, ?, ?, ?)
    """, payload)

    conn.commit()
    conn.close()
    return len(payload)


def upsert_products(user_id: int, df: pd.DataFrame) -> int:
    conn = get_conn()
    cur = conn.cursor()

    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()

    payload = []
    for _, r in df.iterrows():
        payload.append((
            user_id,
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
        ))

    cur.executemany("""
        INSERT OR REPLACE INTO products (
            user_id, sku, product_name, category, supplier,
            lead_time_override, days_to_cover_override, service_level_override, min_stock_override,
            unit_cost, unit_price, promo_min_qty, promo_unit_cost
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, payload)

    conn.commit()
    conn.close()
    return len(payload)


def load_sales(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            date AS Date,
            sku AS SKU,
            quantity AS Quantity,
            stock AS Stock
        FROM sales
        WHERE user_id = ?
    """, conn, params=(user_id,))
    conn.close()

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0.0)
    return df.sort_values(["SKU", "Date"]).reset_index(drop=True)


def load_products(user_id: int) -> pd.DataFrame:
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
        WHERE user_id = ?
    """, conn, params=(user_id,))
    conn.close()

    if df.empty:
        return df

    df["SKU"] = df["SKU"].astype(str).str.strip()
    return df


# =========================================================
# PLAN / TRIAL LOGIC
# =========================================================
def trial_days_left(user: dict) -> int | None:
    if user["role"] == "admin":
        return None

    if user["subscription_status"] == "active":
        return None

    if user["subscription_status"] != "trial":
        return 0

    if not user["trial_start"] or not user["trial_days"]:
        return 0

    started = datetime.fromisoformat(user["trial_start"])
    now = datetime.now(timezone.utc)
    days_passed = (now - started).days
    left = int(user["trial_days"]) - days_passed
    return max(left, 0)


def user_has_access(user: dict) -> bool:
    if user["role"] == "admin":
        return True

    if user["subscription_status"] == "active":
        return True

    if user["subscription_status"] == "trial" and trial_days_left(user) > 0:
        return True

    return False


# =========================================================
# HELPERS
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        elif c in ["promominqty", "promo min qty"]:
            rename_map[col] = "PromoMinQty"
        elif c in ["promounitcost", "promo unit cost"]:
            rename_map[col] = "PromoUnitCost"

    return df.rename(columns=rename_map)


def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    required_cols = {"Date", "SKU", "Quantity", "Stock"}
    missing = required_cols - set(df.columns)

    if missing:
        st.error(f"❌ SALES CSV λείπουν στήλες: {', '.join(sorted(missing))}")
        st.info("Απαιτούνται: Date, SKU, Quantity, Stock")
        st.stop()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        st.error("❌ Η στήλη Date έχει λάθος τιμές.")
        st.stop()

    df["SKU"] = df["SKU"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Stock"] = pd.to_numeric(df["Stock"], errors="coerce").fillna(0.0)

    df = (
        df.groupby(["Date", "SKU"], as_index=False)
        .agg({"Quantity": "sum", "Stock": "last"})
    )

    return df.sort_values(["SKU", "Date"]).reset_index(drop=True)


def validate_products(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    if "SKU" not in df.columns:
        st.error("❌ PRODUCTS CSV πρέπει να έχει SKU.")
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
        ["SKU002", "Vitamin C 1000mg", "Supplements", "Supplier B", 10, 20, 95, 10, "", "", "", ""],
    ], columns=[
        "SKU", "ProductName", "Category", "Supplier",
        "LeadTime", "DaysToCover", "ServiceLevel", "MinStock",
        "UnitCost", "UnitPrice", "PromoMinQty", "PromoUnitCost"
    ])
    return sample.to_csv(index=False).encode("utf-8")


@st.cache_data
def generate_demo_sales(num_skus=80, days=730):
    np.random.seed(42)
    skus = [f"SKU{i:04}" for i in range(1, num_skus + 1)]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    rows = []
    for sku in skus:
        base_lam = np.random.randint(2, 30)
        stock = np.random.randint(100, 600)

        for d in dates:
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

    return pd.DataFrame(rows, columns=["Date", "SKU", "Quantity", "Stock"])


@st.cache_data
def generate_demo_products(num_skus=80):
    categories = ["Pharmacy", "Supplements", "Hygiene", "Grocery"]
    suppliers = ["Supplier A", "Supplier B", "Supplier C"]

    rows = []
    for i in range(1, num_skus + 1):
        sku = f"SKU{i:04}"
        category = categories[(i - 1) % len(categories)]
        supplier = suppliers[(i - 1) % len(suppliers)]

        unit_cost = round(np.random.uniform(1.0, 12.0), 2)
        unit_price = round(unit_cost * np.random.uniform(1.4, 2.2), 2)

        promo_min = 100 if i % 5 == 0 else None
        promo_cost = round(unit_cost * 0.90, 2) if promo_min else None

        rows.append([
            sku, f"Demo Product {i}", category, supplier,
            None, None, None, None,
            unit_cost, unit_price, promo_min, promo_cost
        ])

    return pd.DataFrame(rows, columns=[
        "SKU", "ProductName", "Category", "Supplier",
        "LeadTime", "DaysToCover", "ServiceLevel", "MinStock",
        "UnitCost", "UnitPrice", "PromoMinQty", "PromoUnitCost"
    ])


def current_event_multiplier(row, event_name, event_pct, keyword):
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
# INIT + SESSION
# =========================================================
init_db()

if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

# =========================================================
# FIRST-TIME SETUP
# =========================================================
if get_user_count() == 0:
    st.subheader("🔐 First-time setup")
    st.info("Δημιούργησε τον πρώτο admin χρήστη.")

    with st.form("setup_admin_form"):
        admin_username = st.text_input("Admin username")
        admin_email = st.text_input("Admin email")
        admin_password = st.text_input("Admin password", type="password")
        admin_store = st.text_input("Store name", value="Main Store")
        submitted = st.form_submit_button("Create admin")

        if submitted:
            if not admin_username.strip() or not admin_password.strip() or not admin_store.strip() or not admin_email.strip():
                st.error("Συμπλήρωσε όλα τα πεδία.")
            else:
                ok, msg = create_user(
                    username=admin_username,
                    password=admin_password,
                    store_name=admin_store,
                    email=admin_email,
                    role="admin"
                )
                if ok:
                    st.success("✅ Admin created. Refresh and login.")
                else:
                    st.error(msg)
    st.stop()

# =========================================================
# LOGIN / SIGNUP
# =========================================================
if st.session_state["auth_user"] is None:
    st.subheader("🔑 Login / Sign Up")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state["auth_user"] = user
                    st.rerun()
                else:
                    st.error("Λάθος username ή password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            store_name = st.text_input("Store name")
            submitted_signup = st.form_submit_button("Create account")

            if submitted_signup:
                if not new_username.strip() or not new_password.strip() or not store_name.strip() or not email.strip():
                    st.error("Συμπλήρωσε όλα τα πεδία.")
                elif new_password != confirm_password:
                    st.error("Οι κωδικοί δεν ταιριάζουν.")
                elif len(new_password) < 6:
                    st.error("Ο κωδικός πρέπει να έχει τουλάχιστον 6 χαρακτήρες.")
                else:
                    ok, msg = create_user(
                        username=new_username.strip(),
                        password=new_password,
                        store_name=store_name.strip(),
                        email=email.strip(),
                        role="user"
                    )
                    if ok:
                        st.success("✅ Ο λογαριασμός δημιουργήθηκε. Τώρα μπορείς να κάνεις login.")
                    else:
                        st.error(msg)

    st.markdown(f"""
### Plans
- **Free Trial**: {DEFAULT_TRIAL_DAYS} days
- **Paid Plan**: activated later by admin

### Privacy
Each user sees only:
- their own store
- their own products
- their own sales
- their own forecasts
""")
    st.stop()

# =========================================================
# AUTHENTICATED APP
# =========================================================
user = st.session_state["auth_user"]
user_id = user["id"]

if not user_has_access(user):
    st.error("⛔ Your trial has ended or your subscription is inactive.")
    st.info("Please contact admin to activate a paid plan.")
    if st.button("Logout"):
        st.session_state["auth_user"] = None
        st.rerun()
    st.stop()

days_left = trial_days_left(user)

st.sidebar.success(f"👤 {user['username']}")
st.sidebar.caption(f"🏪 {user['store_name']}")
st.sidebar.caption(f"Role: {user['role']}")
st.sidebar.caption(f"Email: {user['email']}")

if user["role"] != "admin":
    if user["subscription_status"] == "trial":
        st.sidebar.info(f"Trial: {days_left} days left")
    else:
        st.sidebar.success(f"Plan: {user['plan_name']} ({user['subscription_status']})")
else:
    st.sidebar.success("Admin access")

if st.sidebar.button("Logout"):
    st.session_state["auth_user"] = None
    st.rerun()

# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.divider()
st.sidebar.header("⚙️ Inventory Strategy")

use_demo = st.sidebar.checkbox("Use demo sample data", value=False)

default_lead_time = st.sidebar.number_input("Default Lead Time (days)", min_value=1, value=7)
default_service_level = st.sidebar.slider("Default Service Level (%)", min_value=80, max_value=99, value=95)
default_days_to_cover = st.sidebar.number_input("Target Inventory (days of stock)", min_value=1, value=14)
default_min_stock = st.sidebar.number_input("Default Minimum Stock", min_value=0.0, value=0.0)
annual_holding_rate_pct = st.sidebar.number_input("Annual Carrying Cost (%)", min_value=0.0, value=18.0)

st.sidebar.divider()
st.sidebar.subheader("📈 Market / Event Adjustment")

event_name = st.sidebar.selectbox(
    "Current event",
    ["None", "Christmas / Holidays", "Flu / Health Spike", "Tourism Season", "Heatwave", "Custom"]
)
event_pct = st.sidebar.number_input("Event demand adjustment (%)", min_value=-50.0, max_value=200.0, value=0.0)
event_keyword = st.sidebar.text_input("Apply event only to keyword (optional)", value="")
manual_market_growth_pct = st.sidebar.number_input(
    "Recent business growth / decline (%)",
    min_value=-80.0,
    max_value=200.0,
    value=0.0
)

st.sidebar.divider()
st.sidebar.subheader("📤 Uploads")

sales_file = st.sidebar.file_uploader("Upload SALES CSV", type="csv")
products_file = st.sidebar.file_uploader("Upload PRODUCTS CSV (optional)", type="csv")

c1, c2 = st.sidebar.columns(2)
apply_upload = c1.button("✅ Apply Upload")
reset_my_data = c2.button("🧨 Reset My Data")

if reset_my_data:
    st.sidebar.warning("This will delete only YOUR store data.")
    confirm_reset = st.sidebar.checkbox("Confirm my reset")
    if confirm_reset:
        reset_user_data(user_id)
        st.sidebar.success("✅ Your store data was deleted.")
        st.stop()

if apply_upload:
    if use_demo:
        n_sales = upsert_sales(user_id, generate_demo_sales())
        n_products = upsert_products(user_id, generate_demo_products())
        st.success(f"✅ Demo data loaded for your store: {n_sales:,} sales rows, {n_products:,} products.")
    else:
        if sales_file is None:
            st.error("❌ Please upload SALES CSV first.")
            st.stop()

        sales_df = pd.read_csv(sales_file)
        sales_df = validate_sales(sales_df)
        n_sales = upsert_sales(user_id, sales_df)

        if products_file is not None:
            products_df = pd.read_csv(products_file)
            products_df = validate_products(products_df)
            n_products = upsert_products(user_id, products_df)
            st.success(f"✅ SALES uploaded: {n_sales:,} rows | PRODUCTS uploaded: {n_products:,} rows")
        else:
            st.success(f"✅ SALES uploaded: {n_sales:,} rows")

    st.info("If you do not see new data immediately, refresh the page.")

# =========================================================
# LOAD USER DATA
# =========================================================
sales = load_sales(user_id)
products = load_products(user_id)

# =========================================================
# TABS
# =========================================================
tabs = ["📤 Uploads", "📦 Products", "📊 Forecast", "📘 How to Use"]
if user["role"] == "admin":
    tabs.append("👥 Users")

tab_objects = st.tabs(tabs)

tab_uploads = tab_objects[0]
tab_products = tab_objects[1]
tab_forecast = tab_objects[2]
tab_help = tab_objects[3]
tab_users = tab_objects[4] if user["role"] == "admin" else None

# =========================================================
# UPLOADS TAB
# =========================================================
with tab_uploads:
    st.subheader(f"📤 Upload Center — {user['store_name']}")

    st.markdown("""
Upload your sales and product data here.

This application builds a data system over time so it can make better reorder decisions based on:
- your own past sales
- recent demand changes
- seasonality
- current market or event conditions
""")

    a, b = st.columns(2)
    a.download_button(
        "⬇️ Download SALES template.csv",
        data=sales_template_csv(),
        file_name="sales_template.csv",
        mime="text/csv"
    )
    b.download_button(
        "⬇️ Download PRODUCTS template.csv",
        data=products_template_csv(),
        file_name="products_template.csv",
        mime="text/csv"
    )

    st.divider()

    if sales.empty:
        st.warning("No sales data uploaded yet for this store.")
    else:
        st.success(f"Stored sales history for this store: {len(sales):,} rows")
        st.dataframe(sales.tail(20), use_container_width=True)

# =========================================================
# PRODUCTS TAB
# =========================================================
with tab_products:
    st.subheader("📦 Product Catalog")

    if products.empty:
        st.info("No product catalog uploaded yet for this store.")
    else:
        st.dataframe(products, use_container_width=True)

# =========================================================
# FORECAST TAB
# =========================================================
with tab_forecast:
    st.subheader(f"📊 Forecast & Reorder Suggestions — {user['store_name']}")

    if sales.empty:
        st.warning("No sales data found for this store. Upload SALES CSV first.")
        st.stop()

    st.markdown("### Sales data preview")
    st.dataframe(sales.tail(50), use_container_width=True)

    sales = sales.sort_values(["SKU", "Date"]).reset_index(drop=True)
    today = sales["Date"].max()
    current_month = int(today.month)

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

    recent_start = today - pd.Timedelta(days=27)
    recent_df = sales[sales["Date"] >= recent_start]
    recent_avg = recent_df.groupby("SKU")["Quantity"].mean().rename("AvgRecent28")

    prev_start = today - pd.Timedelta(days=55)
    prev_end = today - pd.Timedelta(days=28)
    prev_df = sales[(sales["Date"] >= prev_start) & (sales["Date"] <= prev_end)]
    prev_avg = prev_df.groupby("SKU")["Quantity"].mean().rename("AvgPrev28")

    ly_start = recent_start - pd.Timedelta(days=365)
    ly_end = today - pd.Timedelta(days=365)
    ly_df = sales[(sales["Date"] >= ly_start) & (sales["Date"] <= ly_end)]
    ly_avg = ly_df.groupby("SKU")["Quantity"].mean().rename("AvgSamePeriodLastYear")

    sales["Month"] = sales["Date"].dt.month
    month_avg = sales.groupby(["SKU", "Month"])["Quantity"].mean().rename("MonthAvg").reset_index()
    overall_avg = sales.groupby("SKU")["Quantity"].mean().rename("OverallAvg")
    current_month_avg = month_avg[month_avg["Month"] == current_month][["SKU", "MonthAvg"]].rename(columns={"MonthAvg": "CurrentMonthAvg"})

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

    sku_stats["RecentTrendFactor"] = np.where(
        sku_stats["AvgPrev28"] > 0,
        sku_stats["AvgRecent28"] / sku_stats["AvgPrev28"],
        1.0
    )
    sku_stats["RecentTrendFactor"] = sku_stats["RecentTrendFactor"].clip(lower=0.70, upper=1.50)

    sku_stats["SeasonalityFactor"] = np.where(
        sku_stats["OverallAvg"] > 0,
        sku_stats["CurrentMonthAvg"] / sku_stats["OverallAvg"],
        1.0
    )
    sku_stats["SeasonalityFactor"] = sku_stats["SeasonalityFactor"].clip(lower=0.70, upper=1.50)

    sku_stats["BaseForecastDaily"] = (
        0.45 * sku_stats["AvgRecent28"] +
        0.35 * sku_stats["AvgSamePeriodLastYear"] +
        0.20 * sku_stats["AvgAll"]
    )

    sku_stats["ForecastDaily"] = (
        sku_stats["BaseForecastDaily"] *
        sku_stats["RecentTrendFactor"] *
        sku_stats["SeasonalityFactor"]
    )

    sku_stats["ForecastDaily"] = sku_stats["ForecastDaily"] * (1 + manual_market_growth_pct / 100.0)

    if not products.empty:
        sku_stats = sku_stats.merge(products, on="SKU", how="left")

    sku_stats["EventMultiplier"] = sku_stats.apply(
        lambda r: current_event_multiplier(r, event_name, event_pct, event_keyword),
        axis=1
    )
    sku_stats["ForecastDaily"] = sku_stats["ForecastDaily"] * sku_stats["EventMultiplier"]

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

    z_values = sku_stats["ServiceLevelFinal"].apply(lambda x: get_z(int(x)))
    sku_stats["SafetyStock"] = z_values * sku_stats["StdAll"].fillna(0.0) * np.sqrt(sku_stats["LeadTimeFinal"])
    sku_stats["ReorderPoint"] = (sku_stats["ForecastDaily"] * sku_stats["LeadTimeFinal"]) + sku_stats["SafetyStock"]
    sku_stats["TargetStock"] = (sku_stats["ForecastDaily"] * sku_stats["DaysToCoverFinal"]) + sku_stats["SafetyStock"]

    sku_stats["ReorderPoint"] = np.maximum(sku_stats["ReorderPoint"], sku_stats["MinStockFinal"])
    sku_stats["TargetStock"] = np.maximum(sku_stats["TargetStock"], sku_stats["MinStockFinal"])

    sku_stats["OrderQty"] = (sku_stats["TargetStock"] - sku_stats["CurrentStock"]).clip(lower=0).round().astype(int)

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

    if {"UnitCost", "PromoMinQty", "PromoUnitCost"}.issubset(set(sku_stats.columns)):
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

    critical_cols = [c for c in [
        "SKU", "ProductName", "Category", "CurrentStock",
        "ReorderPoint", "DaysOfCover", "StockoutDays",
        "SuggestedOrderFinal", "PromoWorthIt", "Status"
    ] if c in critical.columns]

    st.dataframe(critical[critical_cols], use_container_width=True)

    st.divider()

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

    preferred_cols = [c for c in [
        "SKU", "ProductName", "Category", "Supplier",
        "CurrentStock", "ForecastDaily", "DaysOfCover", "StockoutDays",
        "ReorderPoint", "TargetStock", "OrderQty", "SuggestedOrderFinal",
        "PromoWorthIt", "NetPromoBenefit",
        "LeadTimeFinal", "ServiceLevelFinal", "DaysToCoverFinal",
        "Status"
    ] if c in filtered.columns]

    display_df = filtered[preferred_cols].rename(columns={
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
# HELP TAB
# =========================================================
with tab_help:
    st.subheader("📘 How to Use")

    st.markdown("""
## What this application does

InventoryAI helps a business understand:
- what products it has
- how fast they sell
- what needs reorder
- what risks stockout
- whether supplier promo offers are worth taking
""")

    st.markdown("""
## Accounts and plans

Each user creates:
- username
- email
- password
- store name

Each user sees only:
- their own store
- their own sales
- their own products
- their own reorder suggestions

### Plans
- **Free Trial**: limited trial access
- **Paid Plan**: activated by admin
""")

    st.markdown("""
## Forecast logic

The application uses:
1. Recent sales trend
2. Historical seasonality
3. Same period last year
4. Manual market / event adjustment
5. Inventory rules like lead time, target stock days, service level and minimum stock
""")

    st.markdown("""
## SALES CSV format
Required columns:
- Date
- SKU
- Quantity
- Stock
""")

    st.markdown("""
## PRODUCTS CSV format (optional)
Optional columns:
- SKU
- ProductName
- Category
- Supplier
- LeadTime
- DaysToCover
- ServiceLevel
- MinStock
- UnitCost
- UnitPrice
- PromoMinQty
- PromoUnitCost
""")

    st.success("No separate PDF is required. The user can read the instructions here.")

# =========================================================
# USERS TAB (ADMIN ONLY)
# =========================================================
if tab_users is not None:
    with tab_users:
        st.subheader("👥 User Management (Admin)")
        st.markdown("Create users, activate paid plans, and manage stores.")

        with st.form("create_user_form"):
            new_username = st.text_input("New username")
            new_email = st.text_input("Email")
            new_password = st.text_input("New password", type="password")
            new_store_name = st.text_input("Store name")
            new_role = st.selectbox("Role", ["user", "admin"], index=0)
            create_btn = st.form_submit_button("Create user")

            if create_btn:
                if not new_username.strip() or not new_password.strip() or not new_store_name.strip() or not new_email.strip():
                    st.error("Συμπλήρωσε όλα τα πεδία.")
                else:
                    ok, msg = create_user(
                        username=new_username,
                        password=new_password,
                        store_name=new_store_name,
                        email=new_email,
                        role=new_role
                    )
                    if ok:
                        st.success("✅ User created.")
                    else:
                        st.error(msg)

        st.divider()
        users_df = list_users()
        st.markdown("### Existing users")
        st.dataframe(users_df, use_container_width=True)

        st.divider()
        st.markdown("### Update user plan / status")

        non_admin_df = users_df[users_df["role"] != "admin"] if not users_df.empty else pd.DataFrame()

        if non_admin_df.empty:
            st.info("No non-admin users available.")
        else:
            selected_user_id = st.selectbox("Choose user id", options=non_admin_df["id"].tolist())
            selected_plan = st.selectbox("Plan", ["trial", "starter", "pro"])
            selected_status = st.selectbox("Subscription status", ["trial", "active", "inactive", "expired"])
            selected_trial_days = st.number_input("Trial days", min_value=0, value=DEFAULT_TRIAL_DAYS)

            if st.button("💾 Update plan / status"):
                update_user_plan(
                    user_id=int(selected_user_id),
                    plan_name=selected_plan,
                    subscription_status=selected_status,
                    trial_days=int(selected_trial_days)
                )
                st.success("✅ User plan/status updated.")

        st.divider()
        st.markdown("### Delete user")
        delete_candidates = users_df[users_df["id"] != user_id] if not users_df.empty else pd.DataFrame()

        if delete_candidates.empty:
            st.info("No deletable users available.")
        else:
            selected_delete_id = st.selectbox(
                "Choose user id to delete",
                options=delete_candidates["id"].tolist()
            )
            if st.button("🗑️ Delete selected user"):
                delete_user(int(selected_delete_id))
                st.success("✅ User deleted.")
