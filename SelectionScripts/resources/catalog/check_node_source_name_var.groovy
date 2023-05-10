/**
 * Script which verifies that the current node runs in the node source specified by the NODE_SOURCE variable
 *
 * If the NODE_SOURCE variable is not defined or empty, the script always returns true
 */

selected = false
NODE_SOURCE_NAME = variables.get("NODE_SOURCE")
if (NODE_SOURCE_NAME) {
    selected = (NODE_SOURCE_NAME.equals(System.getProperty("proactive.node.nodesource")));
} else {
    selected = true
}