/**
 * Script which checks the current node name if it's not empty
 *
 * Arguments:
 * node name
 */

import com.google.common.base.Strings;

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

providedNodeName = args[0]

if (Strings.isNullOrEmpty(providedNodeName)) {
    selected = true;
    return;
} else {
	providedNodeName = providedNodeName.trim()
	println "Node name " + nodename + " (expected :  " + providedNodeName + ")";
	selected = (providedNodeName == nodename)
}
