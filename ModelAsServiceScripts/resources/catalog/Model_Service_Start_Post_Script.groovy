sleep(5000) // wait 5s for the container startup

variables.put("ENDPOINT_MODEL", variables.get("ENDPOINT_" + variables.get("INSTANCE_NAME")))

api_token_endpoint = variables.get("ENDPOINT_MODEL") + "/api/get_token"
println "api_token_endpoint: " + api_token_endpoint

user_name = variables.get("USER_NAME")
println "user: " + user_name

// POST
def post = new URL(api_token_endpoint).openConnection();
post.setRequestMethod("POST")
post.setDoOutput(true)
post.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
post.setRequestProperty("Accept", "text/plain");

HashMap<String, String> params = new HashMap<String, String>();
params.put("user", user_name);
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
println "Response: " + postRC;

if(postRC.equals(200)) {
    TOKEN = post.getInputStream().getText()
    println "TOKEN: " + TOKEN;
    variables.put("SERVICE_TOKEN_PROPAGATED", TOKEN)
}
else {
    println "Error while getting TOKEN"
}

post.disconnect();