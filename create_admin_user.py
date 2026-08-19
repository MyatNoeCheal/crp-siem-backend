"""
Bootstraps the FIRST admin account. Run this once, locally, before the
dashboard's login screen has any account to log into -- after this, that
admin can create further analyst/admin accounts from the dashboard's
Settings tab (Admin Users panel) instead of needing shell access again.

Usage:
    python create_admin_user.py

Safe to re-run: if a user with the given username already exists, it
tells you and exits without changing anything, rather than erroring or
duplicating the account.
"""

import getpass
from database import get_db
import auth

db = get_db()


def main():
    print("=== Smart SIEM: create the first admin account ===\n")
    username = input("Username: ").strip()
    if not username:
        print("Username is required.")
        return

    if auth.get_user_by_username(db, username):
        print(f"\nA user named '{username}' already exists -- nothing to do.")
        print("Log in from the dashboard, or use a different username here.")
        return

    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords did not match -- try again.")
        return
    if len(password) < 8:
        print("Use at least 8 characters.")
        return

    display_name = input("Display name (optional, press Enter to skip): ").strip() or None

    user_id = auth.create_user(db, username, password, role="admin", display_name=display_name)
    print(f"\nCreated admin account '{username}' (id={user_id}).")
    print("You can now log in from the dashboard, and create further")
    print("analyst/admin accounts from Settings -> Admin Users.")


if __name__ == "__main__":
    main()