/**
 * Post script which reads the contents of the localspace/.variables file and use it to update the variables map
 * This post script can be used when someone wants to modify variables from shell tasks (bash or cmd).
 *
 * Simply create the file in the main bash script, e.g.:
 *
 * echo "VAR1=example1" > "$localspace/.variables"
 * echo "VAR2=example2" >> "$localspace/.variables"
 *
 * Then add this post script to automatically propagate the changes in the workflow
 *
 */

new File(localspace, ".variables").eachLine { line ->
    keyvalue = line.split("\\s*=\\s*")
    if (keyvalue.length == 2) {
        variables.put(keyvalue[0], keyvalue[1])
    }
}
