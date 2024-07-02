import base64
import logging
import shutil
import uuid
import pandas as pd
import requests

from io import BytesIO
from os import makedirs
from os.path import join, exists
from PIL import Image
from ast import literal_eval as make_tuple
from duckduckgo_search import DDGS

def raiser(msg): raise Exception(msg)

# Keyword search
search_term = "ants"
# Maximum number of images
query_size = 2
# Get data folder
data_path = 'images'

if 'variables' in locals():
    search_engine = variables.get("SEARCH_ENGINE")
    query_size = int(variables.get("QUERY_SIZE")) if variables.get("QUERY_SIZE") else raiser("QUERY_SIZE not defined!")
    search_term = variables.get('SEARCH_TERM') if variables.get("SEARCH_TERM") else raiser("SEARCH_TERM not defined!")
    data_path = variables.get("DATA_FOLDER") if variables.get("DATA_FOLDER") else raiser("DATA_FOLDER not defined!")
    img_size =  variables.get("IMG_SIZE") if variables.get("IMG_SIZE") else raiser("IMG_SIZE not defined!")

# Get a unique ID
ID = str(uuid.uuid4())

# Get image size
img_size = make_tuple(img_size)

# Create an empty dir
images_path = join(data_path, search_term)
if exists(images_path):
    shutil.rmtree(images_path)
makedirs(images_path)
print("images_path: " + images_path)

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
    return f'<img src="data:image/extension;base64,{image_base64(im)}" height="{img_size[1]}" width="{img_size[0]}">'

def search_bing(query_size, search_term):
    # Bing API config
    subscription_key = "70b641bdd21647089d79c0ab0949ede1"
    search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

    # Bing request
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "license": "public", "imageType": "photo"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:query_size]]

    return thumbnail_urls

def search_duckduckgo(query_size, search_term):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.debug("Using duckduckgo_search library for image search")
    ddgs = DDGS()

    try:
        results = ddgs.images(
            keywords=search_term,
            region="wt-wt",
            safesearch="moderate",
            max_results=query_size,
        )
        thumbnail_urls = [result['image'] for result in results]
        return thumbnail_urls
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return []

# Check host site option
thumbnail_urls = search_duckduckgo(query_size, search_term) \
    if search_engine == 'DuckDuckGo' else search_bing(query_size, search_term)

# Check if thumbnail_urls is valid
if not thumbnail_urls:
    raiser("Failed to retrieve image URLs from DuckDuckGo and Bing.")

# Create an image dataframe for preview
images_df = pd.DataFrame(columns=['Images'])

# Save images results
idx = 1
for i, image_url in enumerate(thumbnail_urls):
    print(i, image_url)
    try:
        image_data = requests.get(image_url)
        image_data.raise_for_status()
        image = Image.open(BytesIO(image_data.content))
        image_path = join(images_path, f"{idx}.jpg")
        image.save(image_path)
        images_df = pd.concat([images_df, pd.DataFrame({'Images': [image_path]})], ignore_index=True)
        idx += 1
    except Exception as e:
        logging.error(f"Failed to download or save image from {image_url}: {e}")
print(images_df)

# Convert dataframe to HTML for preview
result = ''
with pd.option_context('display.max_colwidth', None):
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