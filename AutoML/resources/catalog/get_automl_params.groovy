
params_encoded = variables.get('params_encoded')

// If encoded variables are found
if (params_encoded != null && params_encoded.length() > 0)
{
    println "Found encoded variables:"
    println "params_encoded: " + params_encoded
    byte[] params_decoded = params_encoded.decodeBase64()
    input_variables = new String(params_decoded)
    variables.put('INPUT_VARIABLES', input_variables)
}