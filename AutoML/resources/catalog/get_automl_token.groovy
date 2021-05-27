
token_encoded = variables.get('token_encoded')

// If encoded variables are found
if (token_encoded != null && token_encoded.length() > 0)
{
    println "Found encoded variables:"
    println "token_encoded: " + token_encoded
    byte[] token_decoded = token_encoded.decodeBase64()
    token = new String(token_decoded)
}
else {
    job_id = variables.get("PA_JOB_ID")
    token = '{"_token_id": '+job_id+'}'
}
variables.put('TOKEN', token)
resultMap.put("RESULT_JSON", '{"token": '+token+', "loss": Infinity}')