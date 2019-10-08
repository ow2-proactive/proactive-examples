sleep(5000) // wait 5s for the container startup

variables.put("ENDPOINT_MODEL", variables.get("ENDPOINT_" + variables.get("INSTANCE_NAME")))

ENDPOINT_MODEL = variables.get("ENDPOINT_MODEL")
assert !ENDPOINT_MODEL?.trim() == false : "ENDPOINT_MODEL must be defined!"

API_TOKEN_ENDPOINT = ENDPOINT_MODEL + "/api/get_token"
println "API_TOKEN_ENDPOINT: " + API_TOKEN_ENDPOINT

USER_NAME = variables.get("USER_NAME")
assert !USER_NAME?.trim() == false : "USER_NAME must be defined!"
println "USER_NAME: " + USER_NAME

// POST request
def post = null
try {
    post = new URL(API_TOKEN_ENDPOINT).openConnection();
    post.setRequestMethod("POST")
    post.setDoOutput(true)
    post.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
    post.setRequestProperty("Accept", "text/plain");

    HashMap<String, String> params = new HashMap<String, String>();
    params.put("user", USER_NAME);
    Set set = params.entrySet();
    Iterator i = set.iterator();
    StringBuilder postData = new StringBuilder();
    for (Map.Entry<String, String> param : params.entrySet()) {
        if (postData.length() != 0) {
            postData.append('&');
        }
        postData.append(URLEncoder.encode(param.getKey(), "UTF-8"));
        postData.append('=');
        postData.append(URLEncoder.encode(String.valueOf(param.getValue()), "UTF-8"));
    }
    byte[] postDataBytes = postData.toString().getBytes("UTF-8");
    post.getOutputStream().write(postDataBytes);

    def postRC = post.getResponseCode();
    println "POST response: " + postRC;

    assert postRC.equals(200) == true : "Error while getting TOKEN"

    TOKEN = post.getInputStream().getText()
    println "TOKEN: " + TOKEN;
    assert !TOKEN?.trim() == false : "TOKEN is null or empty!"
    variables.put("SERVICE_TOKEN_PROPAGATED", TOKEN)
}
catch(Exception ex) {
    println(ex)
}
finally {
    if (post != null) {
        post.disconnect();
    }
}