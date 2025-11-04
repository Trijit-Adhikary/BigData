import pyodbc

def connect_with_aad_integrated():
    """Connect using ActiveDirectoryIntegrated authentication"""
    
    server = 'myserver.database.windows.net'
    database = 'mydatabase'
    
    connection_string = f"""
    DRIVER={{ODBC Driver 18 for SQL Server}};
    SERVER={server};
    DATABASE={database};
    Authentication=ActiveDirectoryIntegrated;
    Encrypt=yes;
    TrustServerCertificate=no;
    Connection Timeout=30;
    """
    
    try:
        connection = pyodbc.connect(connection_string)
        print("Successfully connected using ActiveDirectoryIntegrated!")
        
        # Test connection
        cursor = connection.cursor()
        cursor.execute("SELECT SYSTEM_USER, USER_NAME()")
        result = cursor.fetchone()
        print(f"Connected as: {result[0]} | Database User: {result[1]}")
        
        return connection
        
    except pyodbc.Error as e:
        print(f"Connection failed: {e}")
        return None

# Usage
if __name__ == "__main__":
    conn = connect_with_aad_integrated()
    if conn:
        conn.close()
