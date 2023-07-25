import boto3
import json

client = boto3.client("sagemaker-runtime", region_name="ap-southeast-2")

endpoint_name = "Your-endpoint"  # Your endpoint name.
content_type = (
    "application/json"  # The MIME type of the input data in the request body.
)

data = {"text": "this is a test of sagemaker endpoint"}


response = client.invoke_endpoint(
    EndpointName=endpoint_name, ContentType=content_type, Body=json.dumps(data)
)

results = response["Body"].read().decode("utf-8")
print(results)
