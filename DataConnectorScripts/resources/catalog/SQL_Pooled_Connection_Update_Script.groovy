import org.ow2.proactive.scheduler.examples.connectionpooling.*
import groovy.ui.SystemOutputInterceptor
import java.sql.SQLException


// Retrieve RDBMS_NAME variable
RDBMS_NAME = variables.get("RDBMS_NAME")

println("BEGIN Pooled Connection and execute Update on " + RDBMS_NAME + " database")

RDBMS_PROTOCOL = ""
RDBMS_DEFAULT_PORT = ""
RDBMS_DATA_SOURCE_CLASS_NAME = ""

if (RDBMS_NAME.equals("postgresql")){
    init("postgresql", "5432", "org.postgresql.ds.PGSimpleDataSource")
} else if (RDBMS_NAME.equals("mysql")){
    init("mysql", "3306", "com.mysql.cj.jdbc.Driver")
} else if (RDBMS_NAME.equals("greenplum")){
    init("postgresql", "5432", "org.postgresql.ds.PGSimpleDataSource")
} else if (RDBMS_NAME.equals("sqlserver")){
    init("sqlserver", "1433", "com.microsoft.sqlserver.jdbc.SQLServerDataSource")
} else if (RDBMS_NAME.equals("oracle")){
    init("oracle" ,"1521", "oracle.jdbc.pool.OracleDataSource")
} else if (RDBMS_NAME.equals("hsqldb")){
    init("hsqldb:hsql" ,"9001", "org.hsqldb.jdbc.JDBCDataSource")
} else {
    throw new IllegalArgumentException("You must specify a valid RDBMS name in the script arguments amongst postgresql, mysql, greenplum, sqlserver, hsqldb or oracle")
}

host = variables.get("HOST")
if (!host) {
    throw new IllegalArgumentException("ERROR: HOST variable is not provided by the user. Empty value is not allowed.")
}

username = variables.get("USERNAME")
if (!username) {
    throw new IllegalArgumentException("ERROR: USERNAME variable is not provided by the user. Empty value is not allowed.")
}

port = variables.get("PORT")
if (!port) {
    port = RDBMS_DEFAULT_PORT
    println("WARNING: PORT variable is not provided by the user. Using the default value:" + RDBMS_DEFAULT_PORT)
}

database = variables.get("DATABASE")
if (!database) {
    throw new IllegalArgumentException("ERROR: DATABASE variable is not provided by the user. Empty value is not allowed.")
}

SecureJDBCParameters = variables.get("SECURE_JDBC_PARAMETERS")

// This key is used for getting the password from 3rd party credentials.
CREDENTIALS_KEY = RDBMS_NAME + "://" + username + "@" + host + ":" + port
password = credentials.get(CREDENTIALS_KEY).trim()

//Construct the jdbc URL
jdbcUrl = "jdbc:" + RDBMS_PROTOCOL + "://" + host + ":" + port + "/"+ database
//Oracle & SQL Server are particular cases
if(RDBMS_PROTOCOL.equals("oracle")){
    jdbcUrl = "jdbc:oracle:thin:@//" + host + ":" + port + "/" + database
} else if(RDBMS_PROTOCOL.equals("sqlserver")) {
    jdbcUrl = "jdbc:" + RDBMS_PROTOCOL + "://" + host + ":" + port + ";database=" + database + ";encrypt=true;trustServerCertificate=true"
    if (SecureJDBCParameters) {
        jdbcUrl = jdbcUrl + ";" + SecureJDBCParameters
    }
}

interceptor = new SystemOutputInterceptor({ id, str -> print(str); false})
interceptor.start()

if(password){
    dbConnectionDetailsBuilder = new DBConnectionDetails.Builder().jdbcUrl(jdbcUrl).username(username).password(password)
}else{
    dbConnectionDetailsBuilder = new DBConnectionDetails.Builder().jdbcUrl(jdbcUrl).username(username)
}

//Add (optional) data source properties to configure the DB pooled connection
variables.entrySet().each { var ->
    if (var.getKey().startsWith("POOL_")){
        dbConnectionDetailsBuilder.addDataSourceProperty(var.getKey().replace("POOL_", ""), var.getValue())
    }}

//Open the pooled connection to the database
dbConnectionDetails = dbConnectionDetailsBuilder.build()
sqlStatements = variables.get("SQL_STATEMENTS")
if (!sqlStatements){
    throw new IllegalArgumentException("ERROR: SQL_STATEMENTS variable is not provided by the user. Empty value is not allowed.")
}
try{
    dBConnectionPoolsHolder = DBConnectionPoolsHolder.getInstance();
    dBConnectionPoolsHolder.executeUpdate(dbConnectionDetails, sqlStatements)
} catch (Exception e) {
    println org.objectweb.proactive.utils.StackTraceUtil.getStackTrace(e)
    throw e;
}
interceptor.stop()

/**
* This method initializes the default data base properties (port, connection protocol and drivers, etc)
*/
def init(String protocol, String port, String dataSourceClassName){
    RDBMS_PROTOCOL = protocol
    RDBMS_DEFAULT_PORT = port
    RDBMS_DATA_SOURCE_CLASS_NAME = dataSourceClassName
}

println("END Pooled Connection and execute Update on " + RDBMS_NAME + " database")