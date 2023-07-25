## Change project_name to your project name
project_name = "chat-gpt-detector" //put your project name here
region       = "ap-southeast-2" //change region if desired to deploy in another region

## Change instance types amd volume size for SageMaker if desired
training_instance_type  = "ml.m5.xlarge"
inference_instance_type = "ml.m5.xlarge"
volume_size_sagemaker   = 5

## Should not be changed with the current folder structure
handler_path = "../../src/lambda_function"
handler      = "config_lambda.lambda_handler"

