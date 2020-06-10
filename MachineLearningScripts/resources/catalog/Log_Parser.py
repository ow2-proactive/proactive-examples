__file__ = variables.get("PA_TASK_NAME")

if str(variables.get("TASK_ENABLED")).lower() == 'false':
    print("Task " + __file__ + " disabled")
    quit()

print("BEGIN " + __file__)

import bz2
import re
import sys
import time as clock
import uuid
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import wget

# -------------------------------------------------------------
# Get data from the propagated variables
#
TIME_FORMAT = '%H%M%S'
DATE_FORMAT = '%d%m%Y'
PATTERNS_FILE = variables.get("PATTERNS_FILE")
LOG_FILE = variables.get("LOG_FILE")
STRUCTURED_LOG_FILE = variables.get("STRUCTURED_LOG_FILE")

INTERVAL = 10000
LOG_FILE = wget.download(LOG_FILE)
PATTERN_FILE = PATTERNS_FILE
STRUCTERED_LOG_FILE = STRUCTURED_LOG_FILE

# -------------------------------------------------------------
# Detect the different patterns
#
print("Reading the pattern file")
df_patterns = pd.read_csv(PATTERN_FILE, sep=';')
df_columns = pd.Series([''])
table = list()
for index, row in df_patterns.iterrows():
    for e in row[2].split(','):
        if e.strip() != '*':
            table.append(e.strip())
table.append('pattern_id')
myList = list(OrderedDict.fromkeys(table))
df_columns = pd.Series(myList)
print("The different patterns included in the Pattern_file were extracted")

# -------------------------------------------------------------
# Parse raw logs
#
df_structured_logs = pd.DataFrame(columns=df_columns)
print("Processing " + LOG_FILE)
k = 0
t = clock.time()
# variables = list()
my_dict = OrderedDict()
print("Logs patterns matching is in progress")
with open(LOG_FILE) as infile:
    for line in infile:
        k = k + 1
        if k % INTERVAL == 0:
            elapsed_time = clock.time() - t
            print(str(k) + " " + str(elapsed_time) + "sec " + line)
        for index, variable_name in df_columns.iteritems():
            vide = np.nan
            my_dict.__setitem__(variable_name.strip(), vide)
        for index, row in df_patterns.iterrows():
            p = row[1]
            pattern = re.compile(p, re.IGNORECASE)
            m = pattern.match(line)
            if m:
                # print('Match found: ', m.group())
                i = 0
                for e in row[2].split(','):
                    i = i + 1
                    if e.strip() != '*':
                        var = m.group(i)
                        if e.strip() == "date":
                            if len(e.strip()) < 8:
                                if len(var) == 5:
                                    idx = 3
                                elif len(var) == 6:
                                    idx = 4
                                str1_split1 = var[:idx]
                                str1_split2 = var[idx:]
                                tranformed_date = str1_split1 + '20' + str1_split2
                                my_dict.__setitem__(e.strip(), datetime.strptime(tranformed_date, DATE_FORMAT))
                            else:
                                my_dict.__setitem__(e.strip(), datetime.strptime(var, DATE_FORMAT))
                        elif e.strip() == "time":
                            my_dict.__setitem__(e.strip(), datetime.strptime(var.strip(), TIME_FORMAT).time())
                        else:
                            my_dict.__setitem__(e.strip(), repr(var.strip()).strip("0"))
                            my_dict.__setitem__('pattern_id', int(row[0]))
                break
        df_inter = pd.DataFrame([my_dict.values()], columns=df_columns)
        df_structured_logs = df_structured_logs.append(df_inter, ignore_index=True)
# -------------------------------------------------------------

print("All logs were matched")

# -------------------------------------------------------------
# Preview results
#
STRUCTURED_LOG_FILE = STRUCTURED_LOG_FILE.lower()
if STRUCTURED_LOG_FILE.endswith('csv'):
    result = df_structured_logs.to_csv()
    resultMetadata.put("file.extension", ".csv")
    resultMetadata.put("file.name", result + ".csv")
    resultMetadata.put("content.type", "text/csv")
elif STRUCTURED_LOG_FILE.endswith('html'):
    # -------------------------------------------------------------
    # HTML preview style
    #
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
    # -------------------------------------------------------------
    result = df_structured_logs.style.set_table_styles(styles).render().encode('utf-8')
    resultMetadata.put("file.extension", ".html")
    resultMetadata.put("file.name", "output.html")
    resultMetadata.put("content.type", "text/html")
else:
    print('Your data is empty')

# -------------------------------------------------------------
# Save the linked variables
#
df_json_logs = df_structured_logs.to_json(orient='split').encode()
compressed_data = bz2.compress(df_json_logs)

dataframe_id = str(uuid.uuid4())
variables.put(dataframe_id, compressed_data)

print("dataframe id: ", dataframe_id)
print('dataframe size (original):   ', sys.getsizeof(df_json_logs), " bytes")
print('dataframe size (compressed): ', sys.getsizeof(compressed_data), " bytes")

resultMetadata.put("task.name", __file__)
resultMetadata.put("task.dataframe_id", dataframe_id)

print("Finished " + LOG_FILE + " PARSING")
# -------------------------------------------------------------
print("END " + __file__)