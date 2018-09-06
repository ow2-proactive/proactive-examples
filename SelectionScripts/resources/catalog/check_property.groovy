/**
 * Script which verifies that the current node has the given java property defined with correct value
 *
 * Arguments:
 * property name
 * property value
 */

import com.google.common.base.Strings;

if (args.length != 2) {
    println "Incorrect number of arguments, expected 2, received " + args.length;
    selected = false;
    return;
}

propertyName = args[0]
propertyValue = args[1]

if (Strings.isNullOrEmpty(propertyName)) {
    println "Given property name was empty";
    selected = false;
    return;
}
propertyName = propertyName.trim()

if (Strings.isNullOrEmpty(propertyValue)) {
    println "Given property value was empty";
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

println "Value of property " + propertyName + ": " + vmPropValue + " (expected :  " + propertyValue + ")";

selected = (propertyValue == vmPropValue)