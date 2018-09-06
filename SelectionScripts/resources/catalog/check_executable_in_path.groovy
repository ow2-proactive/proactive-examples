/**
 * Script which verifies that the given executable is available in the system PATH and can be executed
 *
 * Arguments:
 * executable name (should not contain the file extension like exe, bat, ...)
 */

import com.google.common.base.Strings;

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

executableName = args[0]

if (Strings.isNullOrEmpty(executableName)) {
    println "Given executable name was empty";
    selected = false;
    return;
}

executableName = executableName.trim()

String systemPath = System.getenv("PATH");
String[] tokens = systemPath.split(File.pathSeparator);
for (String folder : tokens) {
    // Browse each folder
    File directory = new File(folder);

    if (!directory.exists()) {
        continue
    } else if (!directory.isDirectory()) {
        continue
    } else {
        File[] subfiles = directory.listFiles();
        for (int i = 0; i < subfiles.length; i++) {
            // remove file extension
            fileName = subfiles[i].getName()
            if (fileName.contains(".")) {
                fileName = fileName.substring(0, fileName.lastIndexOf("."))
            }
            // check if it matches
            if (fileName.equalsIgnoreCase(executableName)) {
                println("found " + subfiles[i])
                selected = true
                return
            }
        }
    }
}