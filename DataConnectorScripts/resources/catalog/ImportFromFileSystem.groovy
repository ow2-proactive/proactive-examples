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
import javax.ws.rs.core.UriBuilder
import org.apache.commons.io.IOUtils
import java.util.regex.Pattern
import org.apache.commons.vfs2.*
import org.apache.commons.vfs2.auth.*
import org.apache.commons.vfs2.impl.*
import org.apache.commons.vfs2.provider.local.*
import org.apache.commons.vfs2.provider.ftp.FtpFileSystemConfigBuilder
import org.apache.commons.vfs2.provider.ftps.FtpsFileSystemConfigBuilder
import org.apache.commons.vfs2.provider.ftps.FtpsDataChannelProtectionLevel
import org.apache.commons.vfs2.provider.sftp.SftpFileSystemConfigBuilder
import org.objectweb.proactive.extensions.dataspaces.vfs.selector.*
import org.apache.commons.vfs2.provider.sftp.*
import java.net.URI
import java.security.PrivateKey
import java.security.KeyFactory
import java.security.KeyStore
import java.security.GeneralSecurityException
import java.security.cert.*
import java.security.spec.PKCS8EncodedKeySpec
import org.bouncycastle.util.io.pem.PemObject
import org.bouncycastle.util.io.pem.PemReader
import java.io.FileReader
import org.apache.commons.net.io.CopyStreamListener
import org.apache.commons.net.io.CopyStreamEvent
import org.apache.commons.net.io.Util
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode

URI_SCHEME = args[0]

//Set connection parameters and retrieve the SFTP/FTP password
host = variables.get("HOST")
port = variables.get("PORT")
username = variables.get("USERNAME")
password = credentials.get(variables.get("SSH_PASSWORD"))
keyManager = null
optsRemote = new FileSystemOptions()
fsManager = null

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
cut = Boolean.parseBoolean(variables.get("CUT_FILE"))
syncFiles = Boolean.parseBoolean(variables.get("SYNC_FILES"))

// used for cleanup in release()
src = null
remoteSrc = null

//Export file(s) to the SFTP/FTP server
importFiles()
release()

/*
* Create File URI
*
* @param host
* @param port
* @param userName
* @return
*/
def createFileUri(def host, def port, def userName) throws URISyntaxException {
    return new URI(URI_SCHEME, userName, host, (port?.trim()) ? port as int : -1, null, null, null);
}


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
        startUrl = UriBuilder.fromUri(createFileUri(host, port, username)).path(remoteDir).build().toString()
        println "startUrl: " + startUrl
        // localBase can be either a global path or a local relative path in the data space
        if (Paths.get(localBase).isAbsolute()) {
            localDir = localBase
        } else {
            localDir = Paths.get(localspace, localBase).toString()
        }
        // Set remoteSrc for cleanup in release()
        remoteSrc = fsManager.resolveFile(localDir);
        (new File(localDir)).mkdirs()
        remoteFileRoot = fsManager.resolveFile(startUrl, optsRemote)
        // Set src for cleanup in release()
        src = remoteFileRoot
        remoteBasePath = remoteFileRoot.getName()
        children = this.remoteFileRoot.findFiles(new org.objectweb.proactive.extensions.dataspaces.vfs.selector.FileSelector(filePattern))
        children.each { remoteFile ->
            String relativePath = File.separator + remoteBasePath.getRelativeName(remoteFile.getName());
            if (remoteFile.getType() == FileType.FILE && !remoteFile.isContentOpen() && remoteFile.getContent().getSize() > 0) {
                println("Examining remote file " + remoteFile.getName());
                standardPath = new File(localDir, relativePath);
                if(syncFiles && standardPath.exists() && !standardPath.isDirectory()) {
                    println("The file " + standardPath.getName() + " will not be imported from the " + URI_SCHEME + " server as it already exists in the local directory " + localDir)
                } else {
                    localUrl = standardPath.toURI().toURL();
                    println("Standard local path is " + standardPath);
                    LocalFile localFile = (LocalFile) fsManager.resolveFile(localUrl.toString());
                    println("Resolved local file name: " + localFile.getName());
                    createParentFolderAndCopyFile(localFile, remoteFile)
                    if(cut) {
                        println("Deleting remote file " + remoteFile.getName())
                        remoteFile.delete()
                    }
                }
            } else {
                println("Ignoring non-file " + remoteFile.getName());
            }
        }
    } catch (FileSystemException ex) {
        throw new RuntimeException(ex);
    }
}

/**
 * Create the parent folder if it does not exist and copy the file locally
 */
void createParentFolderAndCopyFile(LocalFile localFile, FileObject remoteFile) {
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
    copyThroughStream(remoteFile, localFile)
}


void copyThroughStream(FileObject sourceFile, FileObject destinationFile) throws IOException {
    CopyStreamListener listener = new CopyStreamListener() {
        ProgressPrinter printer = new ProgressPrinter();

        @Override
        public void bytesTransferred(CopyStreamEvent event) {
            /* do nothing */
        }

        @Override
        public void bytesTransferred(long totalBytesTransferred, int bytesTransferred, long streamSize) {
            printer.handleProgress(totalBytesTransferred, streamSize);
        }
    }
    InputStream sourceFileIn = sourceFile.getContent().getInputStream();
    try {
        OutputStream destinationFileOut = destinationFile.getContent().getOutputStream();
        try {
            Util.copyStream(sourceFileIn, destinationFileOut, Util.DEFAULT_COPY_BUFFER_SIZE, sourceFile.getContent().getSize(), listener);
        } finally {
            destinationFileOut.close();
        }
    } finally {
        sourceFileIn.close();
    }
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
 * Initializes the connection to the remote SFTP/FTP server
 * Returns the FileSystemManager instance that manages this connection.
 */
void initializeAuthentication() {
    try {
        fsManager = VFS.getManager();
    } catch (FileSystemException ex) {
        throw new RuntimeException("Failed to get fsManager from VFS", ex);
    }
    if (variables.get("AUTHENTICATION_METHOD") != null && variables.get("AUTHENTICATION_METHOD").equals("SSH_PRIVATE_KEY")){
        passphrase = variables.get("PASSPHRASE")
        sshKey= credentials.get(variables.get("SSH_PRIVATE_KEY"))
        try {
            IdentityProvider bytesIdentityInfo
            if(passphrase != null && !passphrase.isEmpty()) {
                bytesIdentityInfo = new BytesIdentityInfo(sshKey.getBytes(), passphrase.getBytes())
            } else {
                bytesIdentityInfo = new BytesIdentityInfo(sshKey.getBytes())
            }
            //ssh key
            SftpFileSystemConfigBuilder.getInstance().setStrictHostKeyChecking(optsRemote, "no");
            //set root directory to user home
            SftpFileSystemConfigBuilder.getInstance().setUserDirIsRoot(optsRemote, true);
            //timeout
            SftpFileSystemConfigBuilder.getInstance().setTimeout(optsRemote, 10000);
            SftpFileSystemConfigBuilder.getInstance().setIdentityProvider(optsRemote, bytesIdentityInfo)
        } catch (FileSystemException ex) {
            throw new RuntimeException("Failed to set user authenticator", ex);
        }
    } else {
        def auth = new StaticUserAuthenticator(null, username, password)
        try {
            DefaultFileSystemConfigBuilder.getInstance().setUserAuthenticator(optsRemote, auth)
            if (keyManager != null) {
                FtpsFileSystemConfigBuilder.getInstance().setKeyManager(optsRemote, keyManager)
            }
            if(URI_SCHEME.equals("ftps")){
                protectionLevelMap = ["Clear": FtpsDataChannelProtectionLevel.P, "Safe": FtpsDataChannelProtectionLevel.P, "Confidential": FtpsDataChannelProtectionLevel.P, "Private": FtpsDataChannelProtectionLevel.P]
                FtpsFileSystemConfigBuilder.getInstance().setDataChannelProtectionLevel(optsRemote, protectionLevelMap[variables.get("PROTECTION_LEVEL")])
            }
        } catch (FileSystemException ex) {
            throw new RuntimeException("Failed to set user authenticator", ex);
        }
    }
    FtpFileSystemConfigBuilder.getInstance().setPassiveMode(optsRemote, true);
    SftpFileSystemConfigBuilder.getInstance().setDisableDetectExecChannel(optsRemote, true)
}


class ProgressPrinter {
    static final int SIZE_OF_PROGRESS_BAR = 50;
    boolean printedBefore = false;
    BigDecimal progress = new BigDecimal(0);

    void handleProgress(long totalBytesTransferred, long streamSize) {

        BigDecimal numerator = BigDecimal.valueOf(totalBytesTransferred);
        BigDecimal denominator = BigDecimal.valueOf(streamSize);
        BigDecimal fraction = numerator.divide(denominator, new MathContext(2, RoundingMode.HALF_EVEN));
        if (fraction.equals(progress)) {
            /** don't bother refreshing if no progress made */
            return;
        }

        BigDecimal outOfTwenty = fraction.multiply(new BigDecimal(SIZE_OF_PROGRESS_BAR));
        BigDecimal percentage = fraction.movePointRight(2);
        StringBuilder builder = new StringBuilder();
        if (printedBefore) {
            builder.append('\r');
        }

        builder.append("[");
        for (int i = 0; i < SIZE_OF_PROGRESS_BAR; i++) {
            if (i < outOfTwenty.intValue()) {
                builder.append("#");
            } else {
                builder.append(" ");
            }
        }

        builder.append("] ");
        builder.append(percentage.setScale(0, BigDecimal.ROUND_HALF_EVEN).toPlainString()).append("%");

        println(builder);
        // track progress
        printedBefore = true;
        progress = fraction;
    }
}

result = true