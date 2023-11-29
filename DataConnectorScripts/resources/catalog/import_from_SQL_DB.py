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

print("BEGIN importing data from " + DBMS_NAME + " database using " + DBMS_DRIVER + " driver")

HOST = variables.get("HOST")
PORT = variables.get("PORT")
DATABASE = variables.get("DATABASE")
USER = variables.get("USER")

CREDENTIALS_KEY = DBMS_NAME + "://" + USER + "@" + HOST + ":" + str(PORT)
# This key is used for getting the password from 3rd party credentials.
PASSWORD=credentials.get(CREDENTIALS_KEY)

SQL_QUERY = variables.get("SQL_QUERY")
OUTPUT_FILE = variables.get("OUTPUT_FILE")
OUTPUT_TYPE = variables.get("OUTPUT_TYPE")

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
if not SQL_QUERY:
    print("ERROR: SQL_QUERY variable is not provided by the user.")
    sys.exit(1)

print("EXECUTING QUERY...")
print("INSERTING DATA IN {0} DATABASE...".format(DBMS_NAME))
print('HOST= ', HOST)
print('PORT= ', PORT)
print('USER= ', USER)
print('DATABASE= ', DATABASE)
print('QUERY= ', SQL_QUERY)
if OUTPUT_FILE:
    print('OUTPUT_FILE=' + OUTPUT_FILE)

# Please refer to SQLAlchemy doc for more info about database urls.
# http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
if DBMS_NAME == "greenplum":
    DBMS_NAME = "postgresql"
database_url = '{0}+{1}://{2}:{3}@{4}:{5}/{6}'.format(DBMS_NAME,DBMS_DRIVER,USER,PASSWORD,HOST,PORT,DATABASE)
engine = create_engine(database_url)

try:
    with engine.connect() as conn, conn.begin():
        #pd.read_sql() can take either a SQL query as a parameter or a table name
        dataframe = pd.read_sql(SQL_QUERY, conn)

finally:
    conn.close()
    engine.dispose()

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

print("END importing data")