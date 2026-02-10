import sqlite3
import pandas as pd
import os

# Check if database exists
if not os.path.exists('user.db'):
    print("❌ Database not found!")
    print("Please run 'python app.py' first to create the database.")
    exit()

# Connect to database
conn = sqlite3.connect('user.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"Found tables: {[t[0] for t in tables]}\n")

print("=" * 80)
print("REGISTERED USERS")
print("=" * 80)
try:
    users = pd.read_sql_query("SELECT id, username, email, address FROM user", conn)
    print(users.to_string())
    print(f"\nTotal Users: {len(users)}\n")
except Exception as e:
    print(f"No users found: {e}\n")
    users = pd.DataFrame()

print("=" * 80)
print("UPLOADED DOCUMENTS")
print("=" * 80)
try:
    documents = pd.read_sql_query("SELECT * FROM document", conn)
    print(documents.to_string())
    print(f"\nTotal Documents: {len(documents)}\n")
except Exception as e:
    print(f"No documents found: {e}\n")
    documents = pd.DataFrame()

print("=" * 80)
print("CHAT HISTORY")
print("=" * 80)
try:
    chats = pd.read_sql_query("SELECT id, user_id, query, response, created_at FROM chat_history ORDER BY created_at DESC LIMIT 20", conn)
    print(chats.to_string())
    print(f"\nTotal Chats Shown: {len(chats)}\n")
except Exception as e:
    print(f"No chat history found: {e}\n")
    chats = pd.DataFrame()

# Export to Excel
if not users.empty or not documents.empty or not chats.empty:
    try:
        with pd.ExcelWriter('database_export.xlsx') as writer:
            if not users.empty:
                users.to_excel(writer, sheet_name='Users', index=False)
            if not documents.empty:
                documents.to_excel(writer, sheet_name='Documents', index=False)
            if not chats.empty:
                chats.to_excel(writer, sheet_name='Chat_History', index=False)
        print("✅ Data exported to 'database_export.xlsx'")
    except Exception as e:
        print(f"Export failed: {e}")
else:
    print("⚠️ No data to export. Use the application first to generate data.")

conn.close()
