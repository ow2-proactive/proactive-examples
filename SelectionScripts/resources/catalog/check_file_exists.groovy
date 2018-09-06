/**
 * Script which verifies that the given path exists and is readable in the current machine
 *
 * Arguments:
 * file or folder path (use \\ as path delimiter on windows)
 */

import com.google.common.base.Strings;

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

pathName = args[0]

if (Strings.isNullOrEmpty(pathName)) {
    println "Given path name was empty";
    selected = false;
    return;
}

pathName = pathName.trim()

pathFile = new File(pathName);

pathExits = pathFile.exists()
pathReadable = pathFile.canRead()

if (!pathExits) {
    println "Path " + pathName + " does not exists";
    selected = false;
    return;
}

if (!pathReadable) {
    println "Path " + pathName + " exists but is not readable";
    selected = false;
    return;
}

println "Found " + pathFile

selected = true