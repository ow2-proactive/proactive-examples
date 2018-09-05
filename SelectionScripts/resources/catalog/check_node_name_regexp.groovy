/**
 * Script which verifies that the current node name matches the given regular expression
 *
 * Arguments:
 * node name (regexp)
 */

import com.google.common.base.Strings;
import java.util.regex.Pattern;

if (args.length != 1) {
    println "Incorrect number of arguments, expected 1, received " + args.length;
    selected = false;
    return;
}

providedNodeName = args[0]

if (Strings.isNullOrEmpty(providedNodeName)) {
    println "Given node name was empty";
    selected = false;
    return;
}

providedNodeName = providedNodeName.trim()

println "Node name " + nodename + " (expected :  " + providedNodeName + ")";

regex = Pattern.compile(providedNodeName, Pattern.CASE_INSENSITIVE);
regexMatcher = regex.matcher(nodename);

selected = regexMatcher.find()