# Copyright Activeeon 2007-2021. All rights reserved.
__file__ = variables.get("PA_TASK_NAME")

import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import xml.sax.saxutils as saxutils 

if 'variables' in locals():
  PREDICT_DATA = variables.get("PREDICT_DATA_JSON")
  OUTPUT_FILE = variables.get("OUTPUT_FILE")

assert PREDICT_DATA is not None
df = pd.read_json(PREDICT_DATA, orient='split')  

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
  

result = ''
with pd.option_context('display.max_colwidth', -1):
  result = df.to_html(escape=False, formatters=dict(Images=image_formatter, Outputs=image_formatter), classes='table table-bordered table-striped', justify='center')

result = """
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="UTF-8">
                  <title>Deep Learning Preview</title>
                  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
              </head>
                <body class="table text-center">
                	<h2 class="text-center my-3" style="color:#003050;">Deep Learning Results</h2>
                     {0}
                </body></html>
""".format(result)

    
if OUTPUT_FILE == 'HTML':  
    result = result.encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "result.html")
    resultMetadata.put("content.type", "text/html")
else:
  print('It is not possible to preview the HTML format!')

print("END " + __file__)