import java.sql.ResultSet
import org.ow2.proactive.scheduler.examples.connectionpooling.*
import groovy.ui.SystemOutputInterceptor
import java.sql.ResultSetMetaData
import java.sql.SQLException
import org.apache.commons.dbutils.handlers.MapListHandler

// Retrieve the RDBMS name
RDBMS_NAME = variables.get("RDBMS_NAME")

println("BEGIN Pooled Connection and execute Query on " + RDBMS_NAME + " database")

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

// This key is used for getting the password from 3rd party credentials.
CREDENTIALS_KEY = RDBMS_NAME + "://" + username + "@" + host + ":" + port
password = credentials.get(CREDENTIALS_KEY).trim()

//Construct the jdbc URL
jdbcUrl = "jdbc:" + RDBMS_PROTOCOL + "://" + host + ":" + port + "/"+ database
//Oracle & SQL Server are particular cases
if(RDBMS_PROTOCOL.equals("oracle")){
    jdbcUrl = "jdbc:oracle:thin:@//" + host + ":" + port + "/" + database
} else if(RDBMS_PROTOCOL.equals("sqlserver")){
    jdbcUrl = "jdbc:" + RDBMS_PROTOCOL + "://" + host + ":" + port + ";database=" + database
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
ResultSet rs = null
try {
    dBConnectionPoolsHolder = DBConnectionPoolsHolder.getInstance()
    rs = dBConnectionPoolsHolder.executeQuery(dbConnectionDetails, sqlStatements)
} catch (Exception e) {
    println org.objectweb.proactive.utils.StackTraceUtil.getStackTrace(e)
    throw e
}
interceptor.stop()

outputType = variables.get("OUTPUT_TYPE")
outputFile = variables.get("OUTPUT_FILE")
if(!outputFile){
    outputFile = "query-result.csv"
}
if (outputType == "HTML"){
    result = getHtmlPreview(rs).toString().getBytes()
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")
} else {
    csvFile = new File(outputFile) << getCsvPreview(rs).toString()
    result = csvFile.getBytes()
    resultMetadata.put("file.extension", ".csv")
    resultMetadata.put("file.name", outputFile)
    resultMetadata.put("content.type", "text/csv")
}

storeResultVariable = variables.get("STORE_RESULT_VARIABLE")
if(storeResultVariable){
    try {
        handler = new MapListHandler()
        variables.put(storeResultVariable, handler.handle(rs))
    } catch (Exception e) {
        println org.objectweb.proactive.utils.StackTraceUtil.getStackTrace(e)
        throw e
    }
}

/**
 * This method initializes the default data base properties (port, connection protocol and drivers, etc)
 */
def init(String protocol, String port, String dataSourceClassName){
    RDBMS_PROTOCOL = protocol
    RDBMS_DEFAULT_PORT = port
    RDBMS_DATA_SOURCE_CLASS_NAME = dataSourceClassName
}

/**
 * This methods allows to download the results from the Scheduler Portal in a CSV format.
 */
def getCsvPreview(ResultSet rs) throws IOException, SQLException {

    ResultSetMetaData rsmd = rs.getMetaData()
    StringBuilder csvTable  = new StringBuilder()
    int columnCount = rsmd.getColumnCount()
    for (int i=1; i <= columnCount; i++) {
        csvTable.append(rsmd.getColumnName(i))
        if (i != columnCount) {
            csvTable.append(',')
        }
    }
    csvTable.append('\n')
    while (rs.next()) {
        for (int i = 1; i <= columnCount; i++) {
            value = rs.getString(i)
            csvTable.append(value == null ? "" : value.toString())
            if (i != columnCount) {
                csvTable.append(',')
            }
        }
        csvTable.append('\n')
    }
    rs.beforeFirst()
    return csvTable
}

/**
 * This methods allows to preview the results in the Scheduler Portal in a HTML format.
 */
def getHtmlPreview(ResultSet rs) throws IOException, SQLException {

    ResultSetMetaData md = rs.getMetaData()
    StringBuilder htmlTable  = new StringBuilder("<style  type=\"text/css\" >th { font-weight: bold; text-align: center; background: #0B6FA4; color: white;} td {text-align: right;padding: 3px 5px;border-bottom: 1px solid #999999;} table {border: 1px solid #999999;text-align: center;width: 100%;border: 1px solid #999999;}</style>");
    int count = md.getColumnCount()
    htmlTable.append("<table border=1>")
    htmlTable.append("<tr>")
    for (int i=1; i<=count; i++) {
        htmlTable.append("<th>")
        htmlTable.append(md.getColumnLabel(i))
    }
    htmlTable.append("</tr>")
    while (rs.next()) {
        htmlTable.append("<tr>")
        for (int i=1; i<=count; i++) {
            htmlTable.append("<td>")
            htmlTable.append(rs.getString(i))
        }
        htmlTable.append("</tr>")
    }
    htmlTable.append("</table>")
    rs.beforeFirst()
    return htmlTable
}


println("END execute query in " + RDBMS_NAME + " database")