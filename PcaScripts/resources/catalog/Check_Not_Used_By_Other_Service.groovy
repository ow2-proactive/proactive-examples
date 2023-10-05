// deny execution if the node contains any token that starts with PSA_
selected = !nodeTokens.stream().anyMatch { it.startsWith("PSA_") }