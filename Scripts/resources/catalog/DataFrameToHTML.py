import pandas as pd

def dataframe2html(dataframe):
  result = ""
  with pd.option_context('display.max_colwidth', -1):
    result = dataframe.to_html(escape=False)

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
        <meta charset="UTF-8" />
        <style>{0}</style>
      </head>
      <body>{1}</body>
    </html>
    """.format(css_style, result)
    result = result.encode('utf-8')
  return result