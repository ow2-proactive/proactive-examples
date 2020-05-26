import javax.net.ssl.HttpsURLConnection;

sleep(10000) // wait 5s for the container startup

//def ENDPOINT_PATH = "/api/ui"
ENDPOINT_MODEL = variables.get("ENDPOINT_" + variables.get("INSTANCE_NAME")) // -ENDPOINT_PATH
ENDPOINT_MODEL = ENDPOINT_MODEL.split("/api")[0]

variables.put("ENDPOINT_MODEL", ENDPOINT_MODEL)

//ENDPOINT_MODEL = variables.get("ENDPOINT_MODEL")
assert !ENDPOINT_MODEL?.trim() == false : "ENDPOINT_MODEL must be defined!"
assert ENDPOINT_MODEL.startsWith("http") : "ENDPOINT_MODEL should starts with http*!"

HTTPS_ENABLED = false
if (ENDPOINT_MODEL.startsWith("https://")) {
    HTTPS_ENABLED = true
}

API_TOKEN_ENDPOINT = ENDPOINT_MODEL + "/api/get_token"
println "API_TOKEN_ENDPOINT: " + API_TOKEN_ENDPOINT

USER_NAME = variables.get("USER_NAME")
assert !USER_NAME?.trim() == false : "USER_NAME must be defined!"
println "USER_NAME: " + USER_NAME

def nullTrustManager = [
    checkClientTrusted: { chain, authType ->  },
    checkServerTrusted: { chain, authType ->  },
    getAcceptedIssuers: { null }
]

def nullHostnameVerifier = [
    verify: { hostname, session -> true 
    }
]

// POST request
try {
    def post = null
    if (HTTPS_ENABLED) {
        javax.net.ssl.SSLContext sc = javax.net.ssl.SSLContext.getInstance("SSL")
        sc.init(null, [nullTrustManager as javax.net.ssl.X509TrustManager] as javax.net.ssl.X509TrustManager[], null)
        javax.net.ssl.HttpsURLConnection.setDefaultSSLSocketFactory(sc.getSocketFactory())
        HttpsURLConnection.setDefaultHostnameVerifier(nullHostnameVerifier as javax.net.ssl.HostnameVerifier);
        post = (HttpsURLConnection) new URL(API_TOKEN_ENDPOINT).openConnection();
    } else {
        post = new URL(API_TOKEN_ENDPOINT).openConnection();
    }
    
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