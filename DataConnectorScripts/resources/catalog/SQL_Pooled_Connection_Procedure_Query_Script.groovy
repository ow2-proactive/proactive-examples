import java.sql.ResultSet
import org.ow2.proactive.scheduler.examples.connectionpooling.*
import groovy.ui.SystemOutputInterceptor
import java.sql.ResultSetMetaData
import java.sql.SQLException
import org.apache.commons.dbutils.handlers.MapListHandler

// Retrieve the RDBMS name
RDBMS_NAME = variables.get("RDBMS_NAME")

println("BEGIN Pooled Connection and execute stored procedure on " + RDBMS_NAME + " database")

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
storedProcedure = variables.get("STORED_PROCEDURE")
if (!storedProcedure){
    throw new IllegalArgumentException("ERROR: STORED_PROCEDURE variable is not provided by the user. Empty value is not allowed.")
}

ResultSet rs = null
try {
    dBConnectionPoolsHolder = DBConnectionPoolsHolder.getInstance()
    Object[] parsedParams = parseStoredProcedure(storedProcedure);
    String procedureName = storedProcedure.contains("(")
            ? storedProcedure.substring(0, storedProcedure.indexOf('('))
            : storedProcedure;
    rs = dBConnectionPoolsHolder.executeStoredProcedure(dbConnectionDetails, procedureName, parsedParams)
} catch (Exception e) {
    println org.objectweb.proactive.utils.StackTraceUtil.getStackTrace(e)
    throw e
}
interceptor.stop()

outputType = variables.get("OUTPUT_TYPE")
outputFile = variables.get("OUTPUT_FILE")
if(!outputFile){
    outputFile = "procedure-execution-result.csv"
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
 * Parses a stored procedure string to extract the parameters and convert them to appropriate types.
 *
 * @param storedProcedure the stored procedure string to parse, including parameters
 * @return an array of objects representing the parsed parameters
 * @throws IllegalArgumentException if a parameter format is unsupported
 */
def parseStoredProcedure(String storedProcedure) {
    // Extract the parameters within parentheses
    int startIndex = storedProcedure.indexOf('(');
    int endIndex = storedProcedure.lastIndexOf(')');

    if (startIndex == -1 || endIndex == -1) {
        // If no parentheses are found, assume no parameters
        return new Object[0];
    }

    String paramString = storedProcedure.substring(startIndex + 1, endIndex).trim();
    if (paramString.isEmpty()) {
        // No parameters
        return new Object[0];
    }

    String[] paramArray = paramString.split(",\\s*");

    return paramArray.collect { param ->
        param = param.trim()
        if (param ==~ /^\d+$/) { // Integer
            return Integer.parseInt(param)
        } else if (param.equalsIgnoreCase("null")) { // Null
            return null
        } else if (param ==~ /^\d+\.\d+$/) { // Double/Float
            return Double.parseDouble(param)
        } else if (param ==~ /^(?i:true|false)$/) { // Boolean
            return Boolean.parseBoolean(param.toLowerCase())
        } else if (param.startsWith("'")) { // String
            return param.substring(1, param.length() - 1)
        } else if (param ==~ /^\d{4}-\d{2}-\d{2}$/) { // Date in 'YYYY-MM-DD'
            return java.sql.Date.valueOf(param)
        } else if (param ==~ /^\d{2}:\d{2}:\d{2}$/) { // Time in 'HH:MM:SS'
            return java.sql.Time.valueOf(param)
        } else if (param ==~ /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/) { // Timestamp in 'YYYY-MM-DDTHH:MM:SS'
            return java.sql.Timestamp.valueOf(param.replace('T', ' '))
        } else {
            throw new IllegalArgumentException("Unsupported parameter format: " + param)
        }
    } as Object[]

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


println("END execute stored procedure in " + RDBMS_NAME + " database")