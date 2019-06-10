selected = false
NODE_SOURCE_NAME = variables.get("NODE_SOURCE_NAME")
if (NODE_SOURCE_NAME) {
	selected = (System.getProperty("proactive.node.nodesource").equals(NODE_SOURCE_NAME));
} else {
    selected = true
}