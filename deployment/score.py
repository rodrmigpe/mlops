# Deploy a ML model as a Web Serice in Azure ML

import json
import numpy
import joblib
import time
from azureml.core.model import Model


# In[ ]:

# Initialize the model
def init():
    global BEST_MODEL
    # Load the model from file into a global object
    model_path = Model.get_model_path("insurance_model")
    BEST_MODEL = joblib.load(model_path)
    # Print statement for appinsights custom traces:
    print("model initialized" + time.strftime("%H:%M:%S"))


# In[ ]:

# Run Predictions of the new data
def run(raw_data, request_headers):
    # Read the input data from the request in JSON
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    # Predictions of the new data
    result = BEST_MODEL.predict(data)
    # Log the input and output data to Azure Application Insights:
    info = {
        "input": raw_data,
        "output": result.tolist()
        }
    print(json.dumps(info))
  
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

    return {"result": result.tolist()}

