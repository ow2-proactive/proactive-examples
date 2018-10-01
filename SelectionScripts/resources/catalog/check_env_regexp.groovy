/**
 * Script which verifies that the current node has the given environment variable set with a value matching the given regular expression
 *
 * Arguments:
 * environment variable name
 * environment variable value (regexp)
 */

import com.google.common.base.Strings

import java.util.regex.Pattern;

if (args.length != 2) {
    println "Incorrect number of arguments, expected 2, received " + args.length;
    selected = false;
    return;
}

propertyName = args[0]
propertyValue = args[1]

if (Strings.isNullOrEmpty(propertyName)) {
    println "Given environment variable name was empty";
    selected = false;
    return;
}
propertyName = propertyName.trim()

if (Strings.isNullOrEmpty(propertyValue)) {
    println "Given environment variable value was empty";
    selected = false;
    return;
}

propertyValue = propertyValue.trim()

vmPropValue = System.getenv(propertyName);
if (vmPropValue == null) {
    println "Environment variable " + propertyName + " is not defined"
    selected = false;
    return;
}
println "Value of environment variable " + propertyName + ": " + vmPropValue + " (expected :  " + propertyValue + ")";

regex = Pattern.compile(propertyValue, Pattern.CASE_INSENSITIVE);
regexMatcher = regex.matcher(vmPropValue);

selected = regexMatcher.find()