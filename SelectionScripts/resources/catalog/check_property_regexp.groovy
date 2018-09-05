/**
 *  Script which verifies that the current node has the given java property defined and match the given regular expression
 *
 * Arguments:
 * property name
 * property value (regexp)
 */

import com.google.common.base.Strings;
import org.ow2.proactive.scripting.helper.selection.SelectionUtils

if (args.length != 2) {
    println "Incorrect number of arguments, expected 2, received " + args.length;
    selected = false;
    return;
}

propertyName = args[0]
propertyValue = args[1]

if (Strings.isNullOrEmpty(propertyName)) {
    println "Given property Name was empty";
    selected = false;
    return;
}

propertyName = propertyName.trim()

if (Strings.isNullOrEmpty(propertyValue)) {
    println "Given property Value was empty";
    selected = false;
    return;
}

propertyValue = propertyValue.trim()

vmPropValue = System.getProperty(propertyName);
if (vmPropValue == null) {
    println "Java property " + propertyName + " is not defined"
    selected = false;
    return;
}

println "Value of property " + propertyName + ": " + vmPropValue + " (expected : " + propertyValue + ")";

selected = SelectionUtils.checkJavaProperty(propertyName, propertyValue)


