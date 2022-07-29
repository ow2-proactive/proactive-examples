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
import org.bouncycastle.util.io.pem.PemObject
import org.bouncycastle.util.io.pem.PemReader
import java.io.FileReader

URI_SCHEME = args[0]

//Set connection parameters and retrieve the SFTP/FTP password
URL_KEY = URI_SCHEME + "://<username>@<host>";
host = variables.get("HOST")
username = variables.get("USERNAME")
port = variables.get("PORT")
password = checkParametersAndReturnPassword()

//Initialize keystore parameters
clientCertificate = new File(variables.get("CLIENT_CERTIFICATE_FILE_PATH")).text
clientPrivateKeyFile = new File(variables.get("CLIENT_PRIVATE_KEY_FILE_PATH"))
clientPrivateKeyPassword = variables.get("CLIENT_PRIVATE_KEY_PASSWORD")
clientPrivateKeyAlias = variables.get("CLIENT_PRIVATE_KEY_ALIAS")

if(clientCertificate != null & !clientCertificate.isEmpty() && clientPrivateKeyFile != null){
    keyStore = createKeyStore(clientCertificate, clientPrivateKeyFile)
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
exportFiles()
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
def loadPrivateKey(File file) throws Exception {
    KeyFactory factory = KeyFactory.getInstance("RSA");
    FileReader keyReader = null
    PemReader pemReader = null
    println file.getName()
    try {
        keyReader = new FileReader(file)
        pemReader = new PemReader(keyReader)
        PemObject pemObject = pemReader.readPemObject()
        byte[] content = pemObject.getContent()
        PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(content)
        return factory.generatePrivate(privKeySpec)
    } catch (FileSystemException ex) {
        throw new RuntimeException(ex);
    }
}

def createKeyStore(String clientCertificate, File clientPrivateKeyFile) throws IOException, GeneralSecurityException {
    KeyStore keyStore = createEmptyKeyStore()
    X509Certificate publicCert = loadCertificate(new ByteArrayInputStream(IOUtils.toByteArray(clientCertificate)))
    PrivateKey privateKey = loadPrivateKey(clientPrivateKeyFile)
    keyStore.setCertificateEntry("aliasForCertHere", publicCert)
    chain =  [publicCert] as Certificate[]
    keyStore.setKeyEntry(clientPrivateKeyAlias, (PrivateKey)privateKey, clientPrivateKeyPassword.toCharArray(), chain)
    return keyStore
}

/**
 * Retrieves files that match the specified File pattern from the local directory (data space)
 * and export them to the SFTP/FTP server.
 */
void exportFiles() {
    try {
        optsLocal = new FileSystemOptions()
        if (port == null || port.isEmpty()) {
            startRemoteUrl = URI_SCHEME + "://" + host + "/" + remoteDir
        } else {
            startRemoteUrl = URI_SCHEME + "://" + host + ":" + port + "/" + remoteDir
        }
        // Set remoteSrc for cleanup in release()
        remoteSrc = fsManager.resolveFile(startRemoteUrl, optsRemote);
        // localBase can be either a global path or a local relative path in the data space
        if (Paths.get(localBase).isAbsolute()) {
            localDir = localBase
        } else {
            localDir = Paths.get(localspace, localBase).toString()
        }
        startLocalPath = new File(localDir).toURI().toURL().toString();
        println "Local path is : " + startLocalPath
        (new File(localDir)).mkdirs()
        localFileRoot = fsManager.resolveFile(startLocalPath, optsLocal)
        // Set src for cleanup in release()
        src = localFileRoot
        localBasePath = localFileRoot.getName()
        children = localFileRoot.findFiles(new org.objectweb.proactive.extensions.dataspaces.vfs.selector.FileSelector(filePattern))
        children.each { f ->
            String relativePath = localBasePath.getRelativeName(f.getName());
            if (f.getType() == FileType.FILE) {
                println("Examining local file " + f.getName());
                String remoteUrl = startRemoteUrl + "/" + relativePath;
                println("  Remote url is " + remoteUrl);
                FileObject remoteFile = fsManager.resolveFile(remoteUrl, optsRemote);
                println("Resolved remote file name: " + remoteFile.getName());
                createParentFolderAndCopyFile(remoteFile, f)
            } else {
                println("Ignoring non-file " + f.getName());
            }
        }
    } catch (FileSystemException ex) {
        throw new RuntimeException(ex);
    }
}

/**
 * Create the parent folder if it does not exist and copy the file to the remote server
 */
void createParentFolderAndCopyFile(FileObject remoteFile, FileObject f) {
    if (!remoteFile.getParent().exists()) {
        if (!remoteFile.getParent().isWriteable()) {
            throw new RuntimeException("This folder " + remoteFile.getParent() + " is read-only")
        }
        remoteFile.getParent().createFolder();
        println("Create the remote folder " + remoteFile.getParent().toString())
    }
    println("  ### Uploading file ###");
    if (!remoteFile.isWriteable()) {
        throw new RuntimeException("This file " + remoteFile + " is read-only")
    }
    remoteFile.copyFrom(f, new AllFileSelector());
}

/**
 * Release system resources, close connections to the local and remote the filesystems.
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