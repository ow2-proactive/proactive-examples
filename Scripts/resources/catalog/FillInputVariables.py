def fill_input_variables(input_variables):
  for key in input_variables.keys():
    for res in results:
      value = res.getMetadata().get(key)
      if value is not None:
        input_variables[key] = value
        break
  return input_variables