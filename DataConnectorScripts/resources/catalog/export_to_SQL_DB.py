import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys

DBMS_NAME = args[0]
if DBMS_NAME == "PostgreSQL":
    DBMS_DATA=('postgresql','psycopg2',5432,"POSTGRES_")
if DBMS_NAME == "MySQL":
    DBMS_DATA=('mysql','mysqlconnector',3306,"MYSQL_")
if DBMS_NAME == "Greenplum":
    DBMS_DATA=('postgresql','psycopg2',5432,"GPDB_")
if DBMS_NAME == "SQL Server":
    DBMS_DATA=('sqlserver','pyodbc',1433,"SQL_SERVER_")
if DBMS_NAME == "Oracle":
    DBMS_DATA=('oracle','cx_oracle',1521,"ORACLE_")

DBMS_PROTOCOL, DBMS_DRIVER, DBMS_DEFAULT_PORT, DBMS_PREFIX = DBMS_DATA

print("BEGIN Export_Data to " + DBMS_NAME + " database using " + DBMS_DRIVER + " driver")

CREDENTIALS_KEY_MSG = DBMS_PROTOCOL + "://<username>@<host>:<port>"

HOST = variables.get(DBMS_PREFIX + "HOST")
PORT = int(variables.get(DBMS_PREFIX + "PORT"))
DATABASE = variables.get(DBMS_PREFIX + "DATABASE")
USER = variables.get(DBMS_PREFIX + "USER")

# This key is used for getting the password from 3rd party credentials.
CREDENTIALS_KEY = DBMS_PROTOCOL + "://" + USER + "@" + HOST + ":" + str(PORT)
PASSWORD=credentials.get(CREDENTIALS_KEY)

SQL_TABLE = variables.get(DBMS_PREFIX + "TABLE")
INPUT_FILE = variables.get("INPUT_FILE")
INSERT_MODE = variables.get("INSERT_MODE")

if not HOST:
    print("ERROR: {0}HOST variable is not provided by the user.".format(DBMS_PREFIX))
    sys.exit(1)
if not PORT:
    PORT = DBMS_DEFAULT_PORT
    print("WARNING: {0}PORT variable is not provided by the user. Using the default value: {1}".format(DBMS_PREFIX, DBMS_DEFAULT_PORT))
if not DATABASE:
    print("ERROR: {0}DATABASE variable is not provided by the user.".format(DBMS_PREFIX))
    sys.exit(1)
if not USER:
    print("ERROR: {0}USER variable is not provided by the user.".format(DBMS_PREFIX))
    sys.exit(1)
if not PASSWORD:
    print('ERROR: Please add your {0} password to 3rd-party credentials in the scheduler-portal under the key :"{1}"'.format(DBMS_PREFIX,CREDENTIALS_KEY))
    sys.exit(1)
if not INPUT_FILE:
    print("ERROR: INPUT_FILE variable is not provided by the user.")
    sys.exit(1)
if not SQL_TABLE:
    print("ERROR: {0}TABLE variable is not provided by the user.".format(DBMS_PREFIX))
    sys.exit(1)
if not INSERT_MODE:
    INSERT_MODE = 'append'

print("INSERTING DATA IN {0}...".format(DBMS_NAME))
print(DBMS_PREFIX + 'HOST=' , HOST)
print(DBMS_PREFIX + 'PORT=' , PORT)
print(DBMS_PREFIX + 'USER=' , USER)
print(DBMS_PREFIX + 'DATABASE=' , DATABASE)
print(DBMS_PREFIX + 'TABLE=' , SQL_TABLE)

# Please refer to SQLAlchemy doc for more info about database urls.
# http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
database_url = '{0}+{1}://{2}:{3}@{4}:{5}/{6}'.format(DBMS_PROTOCOL,DBMS_DRIVER,USER,PASSWORD,HOST,PORT,DATABASE)
engine = create_engine(database_url)
dataframe = pd.read_csv(INPUT_FILE, sep='\s+|;|,',index_col=None, engine='python')
with engine.connect() as conn, conn.begin():
     dataframe.to_sql(SQL_TABLE, conn, schema=None, if_exists=INSERT_MODE, index=True, index_label=None, chunksize=None, dtype=None)

print("END Export_Data")