selected = false
NODE_SOURCE_NAME = variables.get("NODE_SOURCE")
if (NODE_SOURCE_NAME) {
    selected = (NODE_SOURCE_NAME.equals(System.getProperty("proactive.node.nodesource")));
} else {
    selected = true
}