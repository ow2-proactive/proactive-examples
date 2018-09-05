/**
 * Script which verifies that the current node runs on the given machine (defined by its hostname)
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

if (Strings.isNullOrEmpty(machineName)) {
    println "Given host name was empty";
    selected = false;
    return;
}

machineName = machineName.trim().toLowerCase()

println "Hostname " + nodehost.toLowerCase() + " (expected :  " + machineName + ")";

selected = (nodehost.toLowerCase() == machineName)