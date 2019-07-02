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

print("BEGIN Import_Data from " + DBMS_NAME + " database using " + DBMS_DRIVER + " driver")

CREDENTIALS_KEY_MSG = DBMS_PROTOCOL + "://<username>@<host>:<port>"

HOST = variables.get(DBMS_PREFIX + "HOST")
PORT = int(variables.get(DBMS_PREFIX + "PORT"))
DATABASE = variables.get(DBMS_PREFIX + "DATABASE")
USER = variables.get(DBMS_PREFIX + "USER")

# This key is used for getting the password from 3rd party credentials.
CREDENTIALS_KEY = DBMS_PROTOCOL + "://" + USER + "@" + HOST + ":" + str(PORT)
PASSWORD=credentials.get(CREDENTIALS_KEY)

SQL_QUERY = variables.get(DBMS_PREFIX + "QUERY")
OUTPUT_FILE = variables.get("OUTPUT_FILE")
OUTPUT_TYPE = variables.get("OUTPUT_TYPE")

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
if not SQL_QUERY:
    print("ERROR: {0}QUERY variable is not provided by the user.".format(DBMS_PREFIX))
    sys.exit(1)

print("EXECUTING QUERY...")
print(DBMS_PREFIX + 'HOST=' , HOST)
print(DBMS_PREFIX + 'PORT=' , PORT)
print(DBMS_PREFIX + 'USER=' , USER)
print(DBMS_PREFIX + 'DATABASE=' , DATABASE)
print(DBMS_PREFIX + 'QUERY=' , SQL_QUERY)
if OUTPUT_FILE:
    print('OUTPUT_FILE=' + OUTPUT_FILE)

# Please refer to SQLAlchemy doc for more info about database urls.
# http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urlsdatabase_url = '{0}+{1}://{2}:{3}@{4}:{5}/{6}'.format(DBMS_PROTOCOL,DBMS_DRIVER,USER,PASSWORD,HOST,PORT,DATABASE)
database_url = '{0}+{1}://{2}:{3}@{4}:{5}/{6}'.format(DBMS_PROTOCOL,DBMS_DRIVER,USER,PASSWORD,HOST,PORT,DATABASE)
engine = create_engine(database_url)

with engine.connect() as conn, conn.begin():
    #pd.read_sql() can take either a SQL query as a parameter or a table name
    dataframe = pd.read_sql(SQL_QUERY, conn)
print(dataframe.to_string())
#***************# HTML PREVIEW STYLING #***************#
styles = [
    dict(selector="th", props=[("font-weight", "bold"),
                               ("text-align", "center"),
                               ("background", "#0B6FA4"),
                               ("color", "white")]),
    dict(selector="td", props=[("text-align", "right"),
                               ("padding", "3px 5px"),
                               ("border-bottom", "1px solid #999999")]),
    dict(selector="table", props=[("border", "1px solid #999999"),
                               ("text-align", "center"),
                               ("width", "100%"),
                               ("border", "1px solid #999999")])
]
#******************************************************#

if OUTPUT_TYPE == "HTML":
    print('The task result will be previewed in HTML format')
    result = dataframe.style.set_table_styles(styles).render().encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")
else:
    # Write results to the task result in CSV format
    print('The task result will be written in csv file')
    result = dataframe.to_csv(index=False).encode('utf-8')
    resultMetadata.put("file.extension", ".csv")
    resultMetadata.put("file.name", "result.csv")
    resultMetadata.put("content.type", "text/csv")

# If an OUTPUT_FILE path in the dataspace is designated, then write to this file.
if OUTPUT_FILE:
    dataframe.to_csv(path_or_buf=OUTPUT_FILE, index=False)

print("END Import_Data")