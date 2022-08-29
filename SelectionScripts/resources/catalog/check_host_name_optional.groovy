/**
 * Script which verifies that the current node runs on the given machine (defined by its hostname). If the provided hostname is empty, this script select any node
 *
 * Arguments:
 * machine host name
 */

import com.google.common.base.Strings;

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

machineName = args[0]

if (Strings.nullToEmpty(machineName).trim().isEmpty()) {
    println "Given host name was empty, selection skipped";
    selected = true;
    return;
}

machineName = machineName.trim().toLowerCase()

println "Hostname " + nodehost.toLowerCase() + " (expected :  " + machineName + ")";

selected = (nodehost.toLowerCase() == machineName)