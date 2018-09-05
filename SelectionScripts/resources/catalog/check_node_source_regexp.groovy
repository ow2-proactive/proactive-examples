/**
 * Script which verifies that the current node runs in a node source which matches the specified regular expression
 *
 * Arguments:
 * node source name (regexp)
 */

import com.google.common.base.Strings

import java.util.regex.Pattern;

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

regex = Pattern.compile(nodeSourceName, Pattern.CASE_INSENSITIVE);
regexMatcher = regex.matcher(vmPropValue);

selected = regexMatcher.find()