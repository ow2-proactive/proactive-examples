"""
input_variables = {'task.dataframe_id': None}
fill_input_variables(input_variables)

dataframe_id = input_variables['task.dataframe_id']
print("dataframe id: ", dataframe_id)
"""
def fill_input_variables(input_variables):
  for key in input_variables.keys():
    for res in results:
      value = res.getMetadata().get(key)
      if value is not None:
        input_variables[key] = value
        break
