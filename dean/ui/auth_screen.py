from __future__ import annotations

import streamlit as st

from core.auth import CurrentUser, authenticate_user, create_user, get_permissions, has_users, list_users


def require_login() -> CurrentUser | None:
    if "current_user" in st.session_state:
        user_data = st.session_state["current_user"]
        return CurrentUser(username=user_data["username"], role=user_data["role"])

    if not has_users():
        st.subheader("Create Local Admin")
        st.caption("No cloud login is used. Passwords are hashed and stored locally.")
        username = st.text_input("Admin username")
        password = st.text_input("Admin password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        if st.button("Create admin account"):
            if password != confirm:
                st.error("Passwords do not match.")
                return None
            try:
                create_user(username, password, "Admin")
            except Exception as exc:
                st.error(str(exc))
                return None
            st.success("Admin account created. Sign in to continue.")
        return None

    st.subheader("Local Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        user = authenticate_user(username, password)
        if not user:
            st.error("Invalid username or password.")
            return None
        st.session_state["current_user"] = {"username": user.username, "role": user.role}
        st.rerun()
    return None


def render_user_bar(user: CurrentUser) -> dict[str, bool]:
    permissions = get_permissions(user.role)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.caption(f"Signed in as `{user.username}` with role `{user.role}`")
    with col_b:
        if st.button("Sign out"):
            st.session_state.pop("current_user", None)
            st.rerun()
    return permissions


def render_user_admin() -> None:
    with st.expander("User administration"):
        st.write("Local users")
        st.dataframe(list_users(), use_container_width=True)
        st.write("Create user")
        username = st.text_input("New username", key="new_user_name")
        password = st.text_input("New password", type="password", key="new_user_password")
        role = st.selectbox("Role", ["Viewer", "Editor", "Admin"], key="new_user_role")
        if st.button("Create local user"):
            try:
                create_user(username, password, role)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success("Local user created.")
