/*
#If you want to add more functionalities like a Proxy use
# Please refer to Commons Virtual File System doc for more info.
# https://commons.apache.org/proper/commons-vfs/index.html
*/

import org.apache.commons.net.util.KeyManagerUtils
import org.apache.commons.net.util.TrustManagerUtils
import javax.net.ssl.KeyManager
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import org.apache.commons.io.IOUtils
import java.util.regex.Pattern
import org.apache.commons.vfs2.*
import org.apache.commons.vfs2.auth.*
import org.apache.commons.vfs2.impl.*
import org.apache.commons.vfs2.provider.local.*
import org.apache.commons.vfs2.provider.ftp.FtpFileSystemConfigBuilder
import org.apache.commons.vfs2.provider.ftps.FtpsFileSystemConfigBuilder
import org.objectweb.proactive.extensions.dataspaces.vfs.selector.*
import java.security.PrivateKey
import java.security.KeyFactory
import java.security.KeyStore
import java.security.GeneralSecurityException
import java.security.cert.*
import java.security.spec.PKCS8EncodedKeySpec
import org.bouncycastle.util.io.pem.PemObject
import org.bouncycastle.util.io.pem.PemReader
import java.io.FileReader

URI_SCHEME = args[0]

//Set connection parameters and retrieve the SFTP/FTP password
URL_KEY = URI_SCHEME + "://<username>@<host>";
host = variables.get("HOST")
username = variables.get("USERNAME")
port = variables.get("PORT")
keyManager = null
optsRemote = new FileSystemOptions()
fsManager = null
password = checkParametersAndReturnPassword()

//Initialize keystore parameters
if (variables.get("CLIENT_CERTIFICATE_AUTHENTICATION")) {
    clientCertificate = credentials.get(variables.get("CLIENT_CERTIFICATE_CRED"))
    clientPrivateKey = credentials.get(variables.get("CLIENT_PRIVATE_KEY_CRED"))
    clientPrivateKeyPassword = variables.get("CLIENT_PRIVATE_KEY_PASSWORD")
    clientPrivateKeyAlias = variables.get("CLIENT_PRIVATE_KEY_ALIAS")
    if (clientCertificate != null && !clientCertificate.isEmpty() && clientPrivateKey != null && !clientPrivateKey.isEmpty()) {
        keyStore = createKeyStore(clientCertificate, clientPrivateKey)
        keyManager = KeyManagerUtils.createClientKeyManager(keyStore, clientPrivateKeyAlias, clientPrivateKeyPassword)
    }
}

//Verify server certificate
serverCertificateVerification = variables.get("SERVER_CERTIFICATE_VERIFICATION")
provideServerCertificate = variables.get("PROVIDE_SERVER_CERTIFICATE")
if ("false".equalsIgnoreCase(serverCertificateVerification)) {
    FtpsFileSystemConfigBuilder.getInstance().setTrustManager(optsRemote, TrustManagerUtils.getAcceptAllTrustManager())
}
if ("true".equalsIgnoreCase(provideServerCertificate)) {
    serverCertificate = credentials.get(variables.get("SERVER_CERTIFICATE_CRED"))
    keyStore = createKeyStore(serverCertificate, null)
    FtpsFileSystemConfigBuilder.getInstance().setTrustManager(optsRemote, TrustManagerUtils.getDefaultTrustManager(keyStore))
}

//Initialize the connection manager to the remote SFTP/FTP server.
initializeAuthentication()

//Initialize file pattern, local and remote bases
remoteDir = variables.get("REMOTE_BASE")
filePattern = variables.get("FILE_PATTERN")
localBase = variables.get("LOCAL_BASE")

// used for cleanup in release()
src = null
remoteSrc = null

//Export file(s) to the SFTP/FTP server
importFiles()
release()

def createEmptyKeyStore() throws IOException, GeneralSecurityException {
    KeyStore keyStore = KeyStore.getInstance("JKS")
    keyStore.load(null,null)
    return keyStore
}

/**
 * Load Public Certificate From PEM String
 */
def loadCertificate(InputStream publicCertIn) throws IOException, GeneralSecurityException {
    CertificateFactory factory = CertificateFactory.getInstance("X.509")
    X509Certificate cert = (X509Certificate)factory.generateCertificate(publicCertIn)
    return cert
}

/**
 * Load Private Key From PEM String
 */
def loadPrivateKey(String clientPrivateKey) throws Exception {
    KeyFactory factory = KeyFactory.getInstance("RSA");
    FileReader keyReader = null
    PemReader pemReader = null
    File tmpFile = File.createTempFile("privateKey", ".pem")
    FileWriter writer = new FileWriter(tmpFile)
    writer.write(clientPrivateKey)
    writer.close()
    try {
        keyReader = new FileReader(tmpFile)
        pemReader = new PemReader(keyReader)
        PemObject pemObject = pemReader.readPemObject()
        byte[] content = pemObject.getContent()
        PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(content)
        return factory.generatePrivate(privKeySpec)
    } catch (FileSystemException ex) {
        throw new RuntimeException(ex)
    }
}

def createKeyStore(String certificate, String clientPrivateKey) throws IOException, GeneralSecurityException {
    KeyStore keyStore = createEmptyKeyStore()
    X509Certificate publicCert = loadCertificate(new ByteArrayInputStream(IOUtils.toByteArray(certificate)))
    keyStore.setCertificateEntry("aliasForCertHere", publicCert)
    if(clientPrivateKey != null && !clientPrivateKey.isEmpty()){
        PrivateKey privateKey = loadPrivateKey(clientPrivateKey)
        chain =  [publicCert] as Certificate[]
        keyStore.setKeyEntry(clientPrivateKeyAlias, (PrivateKey)privateKey, clientPrivateKeyPassword.toCharArray(), chain)
    }
    return keyStore
}


/**
 * Retrieves files that match the specified File pattern from the SFTP/FTP server
 * and import them to the local directory (data space).
 */
void importFiles() {
    try {
        if (port == null || port.isEmpty()) {
            startUrl = URI_SCHEME + "://" + host + "/" + remoteDir
        } else {
            startUrl = URI_SCHEME + "://" + host + ":" + port + "/" + remoteDir
        }
        // localBase can be either a global path or a local relative path in the data space
        if (Paths.get(localBase).isAbsolute()) {
            localDir = localBase
        } else {
            localDir = Paths.get(localspace, localBase).toString()
        }
        // Set remoteSrc for cleanup in release()
        remoteSrc = fsManager.resolveFile(localDir);
        (new File(localDir)).mkdirs()
        remoteFile = fsManager.resolveFile(startUrl, optsRemote)
        // Set src for cleanup in release()
        src = remoteFile
        remoteBasePath = remoteFile.getName()
        children = this.remoteFile.findFiles(new org.objectweb.proactive.extensions.dataspaces.vfs.selector.FileSelector(filePattern))
        children.each { f ->
            String relativePath = File.separator + remoteBasePath.getRelativeName(f.getName());
            if (f.getType() == FileType.FILE) {
                println("Examining remote file " + f.getName());
                standardPath = new File(localDir, relativePath);
                localUrl = standardPath.toURI().toURL();
                println("Standard local path is " + standardPath);
                LocalFile localFile = (LocalFile) fsManager.resolveFile(localUrl.toString());
                println("Resolved local file name: " + localFile.getName());
                createParentFolderAndCopyFile(localFile, f)
            } else {
                println("Ignoring non-file " + f.getName());
            }
        }
    } catch (FileSystemException ex) {
        throw new RuntimeException(ex);
    }
}

/**
 * Create the parent folder if it does not exist and copy the file locally
 */
void createParentFolderAndCopyFile(LocalFile localFile, FileObject f) {
    if (!localFile.getParent().exists()) {
        if (!localFile.getParent().isWriteable()) {
            throw new RuntimeException("This folder " + localFile.getParent() + " is read-only")
        }
        localFile.getParent().createFolder();
        println("Create the local folder " + localFile.getParent().toString())
    }
    println("  ### Retrieving file ###");
    if (!localFile.isWriteable()) {
        throw new RuntimeException("This file " + localFile + " is read-only")
    }
    localFile.copyFrom(f, new AllFileSelector());
}

/**
 * Release system resources, close connections to the local and remote filesystems.
 */
void release() {
    FileSystem fs = null;
    FileSystem rfs = null;
    if (src != null) {
        src.close()
        fs = src.getFileSystem() // This works even if the src is closed.
        fsManager.closeFileSystem(fs)
    }
    if (remoteSrc != null) {
        remoteSrc.close()
        rfs = remoteSrc.getFileSystem()
        fsManager.closeFileSystem(rfs)
    }
}

/**
 * Checks whether the provided host, username and port values are empty or not, then
 * returns the SFTP/FTP password using the third party credentials mechanism
 */
def checkParametersAndReturnPassword() {
    if (host.isEmpty()) {
        throw new IllegalArgumentException("HOST variable is not provided by the user. Empty value is not allowed.")
    }
    if (username.isEmpty()) {
        throw new IllegalArgumentException("USERNAME variable is not provided by the user. Empty value is not allowed.")
    }
    def urlKey = URI_SCHEME + "://" + username + "@" + host;
    def password = credentials.get(urlKey)
    if (password == null || password.isEmpty()) {
        throw new IllegalArgumentException("Please add your " + URI_SCHEME + " password to 3rd-party credentials under the key :\"" +
                URL_KEY + "\"");
    }
    return password
}

/**
 * Initializes the connection to the remote SFTP/FTP server
 * Returns the FileSystemManager instance that manages this connection.
 */
void initializeAuthentication() {
    try {
        fsManager = VFS.getManager();
    } catch (FileSystemException ex) {
        throw new RuntimeException("Failed to get fsManager from VFS", ex);
    }
    def auth = new StaticUserAuthenticator(null, username, password)
    try {
        DefaultFileSystemConfigBuilder.getInstance().setUserAuthenticator(optsRemote, auth)
        FtpFileSystemConfigBuilder.getInstance().setPassiveMode(optsRemote, true)
        if (keyManager != null) {
            FtpsFileSystemConfigBuilder.getInstance().setKeyManager(optsRemote, keyManager)
        }

    } catch (FileSystemException ex) {
        throw new RuntimeException("Failed to set user authenticator", ex);
    }
}

result = true