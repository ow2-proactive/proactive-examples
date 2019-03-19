print("BEGIN Preview_Results")

import base64
import pandas as pd
from PIL import Image
from io import BytesIO

if 'variables' in locals():
  PREDICT_DATA = variables.get("PREDICT_DATA_JSON")
  OUTPUT_FILE = variables.get("OUTPUT_FILE")

assert PREDICT_DATA is not None
df = pd.read_json(PREDICT_DATA, orient='split')  

# check the predictions
if {'Predictions','Targets'}.issubset(df.columns):
	pred_result =[]
	for indice in range(len(df)):
		if df['Predictions'][indice] == df['Targets'][indice]:
			result = 'https://github.com/ow2-proactive/automation-dashboard/blob/master/app/styles/patterns/img/wf-icons/tick_green.png?raw=true'
			pred_result.append(result)
		else:
			result = 'https://github.com/ow2-proactive/automation-dashboard/blob/master/app/styles/patterns/img/wf-icons/close_red.png?raw=true'
			pred_result.append(result)
	df_pred_image_url = pd.DataFrame(pred_result)
	df['Results'] = df_pred_image_url
 
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
  return """<img src="{0}" height="50" width="50"/>""".format(im_url)
  

result = ''
with pd.option_context('display.max_colwidth', -1):
  result = df.to_html(escape=False, formatters=dict(Images=image_formatter, Outputs=image_formatter, Results=image_formatter_url))

css_style="""
table {
  border: 1px solid #999999;
  text-align: center;
  border-collapse: collapse;
  width: 100%; 
}
td {
  border: 1px solid #999999;         
  padding: 3px 2px;
  font-size: 13px;
  border-bottom: 1px solid #999999;
  #border-bottom: 1px solid #FF8C00;  
  border-bottom: 1px solid #0B6FA4;   
}
th {
  font-size: 17px;
  font-weight: bold;
  color: #FFFFFF;
  text-align: center;
  background: #0B6FA4;
  #background: #E7702A;       
  #border-left: 2px solid #999999
  border-bottom: 1px solid #FF8C00;            
}
"""
result = """
     
            
            
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="UTF-8">
                  <style>{0}</style>
                </head>
                <body>{1}</body></html>
""".format(css_style, result)

if OUTPUT_FILE == 'HTML':  
    result = result.encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "result.html")
    resultMetadata.put("content.type", "text/html")
else:
  print('It is not possible to preview the HTML format!')

print("END Preview_Results")