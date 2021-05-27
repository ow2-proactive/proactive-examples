// Copyright Activeeon 2007-2021. All rights reserved.
params_encoded = variables.get('params_encoded')
token_encoded = variables.get('token_encoded')

// If encoded variables are found
if ((params_encoded != null && params_encoded.length() > 0) &&
    (token_encoded != null && token_encoded.length() > 0))
{
    println "Found encoded variables:"
    println "params_encoded: " + params_encoded
    println "token_encoded: " + token_encoded
    
    byte[] params_decoded = params_encoded.decodeBase64()
    byte[] token_decoded = token_encoded.decodeBase64()
    
    input_variables = new String(params_decoded)
    token = new String(token_decoded)
    
    variables.put('INPUT_VARIABLES', input_variables)
}
else {
    job_id = variables.get("PA_JOB_ID")
    token = '{"_token_id": '+job_id+'}'
}

variables.put('TOKEN', token)
resultMap.put("RESULT_JSON", '{"token": '+token+', "loss": Infinity}')