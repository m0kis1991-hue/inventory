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
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            store_name = st.text_input("Store name")
            submitted_signup = st.form_submit_button("Create account")

            if submitted_signup:
                if not new_username.strip() or not new_password.strip() or not store_name.strip():
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
                        role="user"
                    )
                    if ok:
                        st.success("✅ Ο λογαριασμός δημιουργήθηκε. Τώρα μπορείς να κάνεις login.")
                    else:
                        st.error(msg)

    st.markdown("""
### Τι σημαίνει αυτό το σύστημα
Κάθε χρήστης βλέπει **μόνο τα δικά του δεδομένα**:
- το δικό του κατάστημα
- τα δικά του προϊόντα
- τις δικές του πωλήσεις
- τις δικές του προβλέψεις
""")
    st.stop()
