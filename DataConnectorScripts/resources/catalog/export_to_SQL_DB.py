import pandas as pd
from sqlalchemy import create_engine
import sys

DBMS_NAME = args[0]
if DBMS_NAME == "postgresql":
    DBMS_DATA=('psycopg2', 5432)
if DBMS_NAME == "mysql":
    DBMS_DATA=('mysqlconnector', 3306)
if DBMS_NAME == "greenplum":
    DBMS_DATA=('psycopg2', 5432)
if DBMS_NAME == "sqlserver":
    DBMS_DATA=('pyodbc', 1433)
if DBMS_NAME == "oracle":
    DBMS_DATA=('cx_oracle', 1521)

DBMS_DRIVER, DBMS_DEFAULT_PORT = DBMS_DATA

print("BEGIN exporting data to " + DBMS_NAME + " database using " + DBMS_DRIVER + " driver")

HOST = variables.get("HOST")
PORT = variables.get("PORT")
DATABASE = variables.get("DATABASE")
USER = variables.get("USER")

CREDENTIALS_KEY = DBMS_NAME + "://" + USER + "@" + HOST + ":" + str(PORT)
# This key is used for getting the password from 3rd party credentials.
PASSWORD=credentials.get(CREDENTIALS_KEY)

SQL_TABLE = variables.get("TABLE")
INPUT_FILE = variables.get("INPUT_FILE")
INSERT_MODE = variables.get("INSERT_MODE")

if not HOST:
    print("ERROR: HOST variable is not provided by the user.")
    sys.exit(1)
if not PORT:
    PORT = DBMS_DEFAULT_PORT
    print("WARNING: PORT variable is not provided by the user. Using the default value: {0}".format(DBMS_DEFAULT_PORT))
if not DATABASE:
    print("ERROR: DATABASE variable is not provided by the user.")
    sys.exit(1)
if not USER:
    print("ERROR: USER variable is not provided by the user.")
    sys.exit(1)
if not PASSWORD:
    print('ERROR: Please add your database password to 3rd-party credentials in the scheduler-portal under the key :"{0}"'.format(CREDENTIALS_KEY))
    sys.exit(1)
if not INPUT_FILE:
    print("ERROR: INPUT_FILE variable is not provided by the user.")
    sys.exit(1)
if not SQL_TABLE:
    print("ERROR: TABLE variable is not provided by the user.")
    sys.exit(1)
if not INSERT_MODE:
    INSERT_MODE = 'append'

print("INSERTING DATA IN {0} DATABASE...".format(DBMS_NAME))
print('HOST= ', HOST)
print('PORT= ', PORT)
print('USER= ', USER)
print('DATABASE= ', DATABASE)
print('TABLE= ', SQL_TABLE)

# Please refer to SQLAlchemy doc for more info about database urls.
# http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
if DBMS_NAME == "greenplum":
    DBMS_NAME = "postgresql"
database_url = '{0}+{1}://{2}:{3}@{4}:{5}/{6}'.format(DBMS_NAME,DBMS_DRIVER,USER,PASSWORD,HOST,PORT,DATABASE)
engine = create_engine(database_url)
dataframe = pd.read_csv(INPUT_FILE, sep='\s+|;|,',index_col=None, engine='python')

try:
    with engine.connect() as conn, conn.begin():
        dataframe.to_sql(SQL_TABLE, conn, schema=None, if_exists=INSERT_MODE, index=True, index_label=None, chunksize=None, dtype=None)

finally:
    conn.close()
    engine.dispose()

print("END exporting data")