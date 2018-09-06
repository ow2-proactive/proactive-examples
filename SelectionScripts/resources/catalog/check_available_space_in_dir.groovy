/**
 * Script which verifies the available disk space in the provided folder path
 *
 * Arguments:
 * folder path (use \\ as path delimiter on Windows)
 * required available space (in megabytes)
 */

import com.google.common.base.Strings;
import java.lang.management.ManagementFactory;

if (args.length != 2) {
    println "Incorrect number of arguments, expected 2, received " + args.length;
    selected = false;
    return;
}

folderPath = args[0]

if (Strings.isNullOrEmpty(folderPath)) {
    println "Given folder path was empty";
    selected = false;
    return;
}

folderPathFile = new File(folderPath.trim())

if (!folderPathFile.exists() || !folderPathFile.isDirectory() || !folderPathFile.canWrite()) {
    println "Folder path " + folderPathFile + " does not exist, is not a directory or is not writeable.";
    selected = false;
    return;
}


requiredSpace = args[1]

if (Strings.isNullOrEmpty(requiredSpace)) {
    println "Given required space was empty";
    selected = false;
    return;
}

requiredSpace = Double.parseDouble(requiredSpace.trim())

MEGABYTE = (1024L * 1024L);

freeSpace = ((double) folderPathFile.getUsableSpace()) / MEGABYTE;


println "Available space in " + folderPathFile + ": " + freeSpace  + "MB (required >= " + requiredSpace + ")";

selected = (freeSpace >= requiredSpace)


