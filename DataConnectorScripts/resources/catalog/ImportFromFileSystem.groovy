/*
#If you want to add more functionalities like a Proxy use
# Please refer to Commons Virtual File System doc for more info.
# https://commons.apache.org/proper/commons-vfs/index.html
*/

import org.apache.commons.net.util.KeyManagerUtils
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

URI_SCHEME = args[0]

///Set connection parameters and retrieve the SFTP/FTP password
URL_KEY = URI_SCHEME + "://<username>@<host>";
host = variables.get("HOST")
username = variables.get("USERNAME")
port = variables.get("PORT")
password = checkParametersAndReturnPassword()

//Initialize keystore parameters
clientCertificate = new File(variables.get("CLIENT_CERTIFICATE_FILE_PATH")).text
clientPrivateKey = new File(variables.get("CLIENT_PRIVATE_KEY_FILE_PATH")).text
clientPrivateKeyPassword = variables.get("CLIENT_PRIVATE_KEY_PASSWORD")
clientPrivateKeyAlias = variables.get("CLIENT_PRIVATE_KEY_ALIAS")

if(clientCertificate != null & !clientCertificate.isEmpty() && clientPrivateKey != null && !clientPrivateKey.isEmpty()){
    keyStore = createKeyStore(clientCertificate, clientPrivateKey)
    keyManager = KeyManagerUtils.createClientKeyManager(keyStore, clientPrivateKeyAlias, clientPrivateKeyPassword)
}

//Initialize the connection manager to the remote SFTP/FTP server.
optsRemote = new FileSystemOptions()
fsManager = null
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
def loadPrivateKey(InputStream privateKeyIn) throws IOException, GeneralSecurityException {
    //need the full file - org.apache.commons.io.IOUtils is handy
    byte[] fullFileAsBytes = IOUtils.toByteArray(privateKeyIn)
    //remember this is supposed to be a text source with the BEGIN/END and base64 in the middle of the file
    String fullFileAsString = new String(fullFileAsBytes)
    //extract out between BEGIN/END
    String encoded = fullFileAsString
            .replace("-----BEGIN PRIVATE KEY-----", "")
            .replaceAll(System.lineSeparator(), "")
            .replace("-----END PRIVATE KEY-----", "")
    //decode the Base64 string
    byte[] keyDecoded = Base64.getMimeDecoder().decode(encoded)
    //for my example, the source is in common PKCS#8 format
    PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(keyDecoded)
    //from there we can use the KeyFactor to generate
    KeyFactory keyFactory = KeyFactory.getInstance("RSA")
    PrivateKey privateKey = keyFactory.generatePrivate(keySpec)
    return privateKey
}

def createKeyStore(String clientCertificate, String clientPrivateKey) throws IOException, GeneralSecurityException {
    KeyStore keyStore = createEmptyKeyStore()
    X509Certificate publicCert = loadCertificate(new ByteArrayInputStream(IOUtils.toByteArray(clientCertificate)))
    PrivateKey privateKey = loadPrivateKey(new ByteArrayInputStream(clientPrivateKey.getBytes()))
    keyStore.setCertificateEntry("aliasForCertHere", publicCert)
    chain =  [publicCert] as Certificate[]
    keyStore.setKeyEntry(clientPrivateKeyAlias, (PrivateKey)privateKey, clientPrivateKeyPassword.toCharArray(), chain)
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
        DefaultFileSystemConfigBuilder.getInstance().setUserAuthenticator(optsRemote, auth);
        if (keyManager != null) {
            FtpFileSystemConfigBuilder.getInstance().setPassiveMode(optsRemote, true);
        } else {
            FtpsFileSystemConfigBuilder.getInstance().setKeyManager(optsRemote, keyManager)
        }
    } catch (FileSystemException ex) {
        throw new RuntimeException("Failed to set user authenticator", ex);
    }
}

result = true