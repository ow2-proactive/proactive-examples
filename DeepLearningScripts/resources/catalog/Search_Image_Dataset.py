import re
import time
import uuid
import json
import shutil
import base64
import logging
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from os.path import join, exists
from os import makedirs

def raiser(msg): raise Exception(msg)

def get_thumbnail(path):
    i = Image.open(path)
    extension = i.format
    i.thumbnail((200, 200), Image.LANCZOS)
    return i, extension

def image_base64(im):
    if isinstance(im, str):
        im, extension = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, extension)
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    extension = im.format
    return f'<img src="data:image/extension;base64,{image_base64(im)}" height="200" width="200">'

def image_formatter_url(im_url):
    return """<img src="{0}" height="100" width="100"/>""".format(im_url)

def variables_get(name, default_value=None):
    if 'variables' in locals():
        if variables.get(name) is not None:
            return variables.get(name)
        else:
            return default_value
    else:
        return default_value

# Keyword search
search_term = "ants" 
# Maximum number of images
query_size = 2 
# Get data folder
DATA_FOLDER = 'images'

if 'variables' in locals():
    host_site   = variables.get("HOST_SITE") 
    query_size  = int(variables.get("QUERY_SIZE")) if variables.get("QUERY_SIZE") else raiser("QUERY_SIZE not defined!")
    search_term = variables.get('SEARCH_TERM') if variables.get("SEARCH_TERM") else raiser("SEARCH_TERM not defined!")
    DATA_FOLDER = variables.get("DATA_FOLDER") if variables.get("DATA_FOLDER") else raiser("DATA_FOLDER not defined!")

# Get an unique ID
ID = str(uuid.uuid4())

# Create an empty dir
images_path = join(DATA_FOLDER, search_term)
if exists(images_path):
    shutil.rmtree(images_path)
makedirs(images_path)
print("images_path: " + images_path)


# search image from bing navigator
def search_bing(query_size, search_term):
	# Bing API config
	# https://docs.microsoft.com/en-us/azure/cognitive-services/bing-image-search/quickstarts/python
	subscription_key = "70b641bdd21647089d79c0ab0949ede1"
	search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

	# Bing request
	headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
	params  = {"q": search_term, "license": "public", "imageType": "photo"}
	response = requests.get(search_url, headers=headers, params=params)
	response.raise_for_status()
	search_results = response.json()
	thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:query_size]]
    
	return thumbnail_urls
    
# Search image from DuckDuckGo navigator    
def search_DuckDuckGo(query_size, search_term):
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    url = 'https://duckduckgo.com/';
    thumbnail_image = []
    list_size =  query_size -1
    
    params = {
    	'q': search_term,
    };

    logger.debug("Hitting DuckDuckGo for Token");

    #   First make a request to above URL, and parse out the 'vqd'
    #   This is a special token, which should be used in the subsequent request
    res = requests.post(url, data=params)
    searchObj = re.search(r'vqd=([\d-]+)\&', res.text, re.M|re.I);

    if not searchObj:
        logger.error("Token Parsing Failed !");
        return -1;

    logger.debug("Obtained Token");

    headers = {
        'authority': 'duckduckgo.com',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'sec-fetch-dest': 'empty',
        'x-requested-with': 'XMLHttpRequest',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'referer': 'https://duckduckgo.com/',
        'accept-language': 'en-US,en;q=0.9',
    }

    params = (
        ('l', 'us-en'),
        ('o', 'json'),
        ('q', search_term),
        ('vqd', searchObj.group(1)),
        ('f', ',,,'),
        ('p', '1'),
        ('v7exp', 'a'),
    )

    requestUrl = url + "i.js";

    logger.debug("Hitting Url : %s", requestUrl)

    while True:
        while True:
            try:
                res = requests.get(requestUrl, headers=headers, params=params)
                data = json.loads(res.text);
                break;
            except ValueError as e:
                logger.debug("Hitting Url Failure - Sleep and Retry: %s", requestUrl)
                time.sleep(5);
                continue;

        logger.debug("Hitting Url Success : %s", requestUrl)

        if list_size < query_size:
            thumbnail_image += data["results"] 
        else: break
        
        
        if "next" not in data:
            logger.debug("No Next Page - Exiting")
            exit(0);
            
        list_size = len(thumbnail_image)
        requestUrl = url + data["next"]
        thumbnail_image[0: query_size]
        thumbnail_urls = [i['thumbnail'] for i in thumbnail_image if 'thumbnail' in i]
        
    return thumbnail_urls

# Check host site option 
thumbnail_urls = search_DuckDuckGo(query_size, search_term) if host_site == 'DuckDuckGo' else search_bing(query_size, search_term)

# Create a image dataframe for preview
images_df = pd.DataFrame(columns = ['Images'])

# Save images results
idx = 1
for i in range(len(thumbnail_urls)):
    image_url = thumbnail_urls[i]
    print(i, image_url)
    time.sleep(3)
    image_data = requests.get(image_url)
    image_data.raise_for_status()
    image = Image.open(BytesIO(image_data.content))
    image_path = join(images_path, str(idx) + ".jpg")
    image.save(image_path)
    images_df = images_df.append({'Images': image_path}, ignore_index=True)
    idx = idx + 1
print(images_df)

# Convert dataframe to HTML for preview
result = ''
with pd.option_context('display.max_colwidth', -1):
    result = images_df.to_html(
        escape=False, 
        formatters=dict(Images=image_formatter), 
        classes='table table-bordered table-striped', 
        justify='center')

result = """
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Images Preview</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
</head>
<body class="container">
<h1 class="text-center my-4" style="color:#003050;">Images Preview</h1>
<div style="text-align:center">{0}</div>
</body></html>
""".format(result)

result = result.encode('utf-8')
if 'resultMetadata' in locals():
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "preview.html")
    resultMetadata.put("content.type", "text/html")