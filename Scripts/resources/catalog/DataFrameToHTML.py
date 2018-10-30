"""
import numpy as np
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
print(df.head())

df_html = dataframe2html(df)
print(df_html)
"""
import pandas as pd

def dataframe2html(dataframe):
  result = ""
  #============================== Preview results ===============================
  #***************# HTML PREVIEW STYLING #***************#
  styles = [
      dict(selector="th", props=[("font-weight", "bold"),
                                 ("text-align", "center"),
                                 ("font-size", "15px"),
                                 ("background", "#0B6FA4"),
                                 ("color", "#FFFFFF")]),
                                 ("padding", "3px 7px"),
      dict(selector="td", props=[("text-align", "right"),
                                 ("padding", "3px 3px"),
                                 ("border", "1px solid #999999"),
                                 ("font-size", "13px"),
                                 ("border-bottom", "1px solid #0B6FA4")]),
      dict(selector="table", props=[("border", "1px solid #999999"),
                                 ("text-align", "center"),
                                 ("width", "100%"),
                                 ("border-collapse", "collapse")])
  ]
  #******************************************************#

  with pd.option_context('display.max_colwidth', -1):
    result = dataframe.style.set_table_styles(styles).render().encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")
  #==============================================================================
  return result
