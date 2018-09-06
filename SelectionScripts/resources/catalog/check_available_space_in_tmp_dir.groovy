/**
 * Script which verifies the available disk space in temp directory
 *
 * Arguments:
 * required available space (in megabytes)
 */

import com.google.common.base.Strings;
import java.lang.management.ManagementFactory;

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

requiredSpace = args[0]

if (Strings.isNullOrEmpty(requiredSpace)) {
    println "Given required space was empty";
    selected = false;
    return;
}

requiredSpace = Double.parseDouble(requiredSpace.trim())

MEGABYTE = (1024L * 1024L);

tmpDir = System.getProperty("java.io.tmpdir")

tmpDirFile = new File(tmpDir)

freeSpace = ((double) tmpDirFile.getUsableSpace()) / MEGABYTE;


println "Available space in " + tmpDir + ": "+ freeSpace + "MB (required >= " + requiredSpace + ")";

selected = (freeSpace >= requiredSpace)


