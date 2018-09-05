/**
 * Script which verifies that the current node runs on a host which matches the given regular expression
 *
 * Arguments:
 * machine host name (regexp)
 */

import com.google.common.base.Strings;
import org.ow2.proactive.scripting.helper.selection.SelectionUtils

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

machineName = machineName.trim()

println "Hostname " + nodehost + " (expected :  " + machineName + ")";

selected = SelectionUtils.checkHostName(machineName)


