/**
 * Script which verifies that the current node runs in the specified node source
 *
 * Arguments:
 * node source name
 */

import com.google.common.base.Strings;

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

nodeSourceName = args[0]

if (Strings.isNullOrEmpty(nodeSourceName)) {
    println "Given node source name was empty";
    selected = false;
    return;
}
nodeSourceName = nodeSourceName.trim()

vmPropValue = System.getProperty("proactive.node.nodesource");

if (vmPropValue == null) {
    // if the node source property is not defined, set it as "Default"
    vmPropValue = "Default";
}

println "Value of property " + "proactive.node.nodesource" + ": " + vmPropValue + " (expected :  " + nodeSourceName + ")";

selected = (nodeSourceName == vmPropValue)