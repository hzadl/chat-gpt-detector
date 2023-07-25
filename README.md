# ChatGPT detector using Huggingface GPT2ForSequenceClassification model

ChatGPT has gained significant popularity, it is very challenging for human to identify ChatGPT generated content.

In this repository, we implement a classifer using the Huggingface GPT2ForSequenceClassification model to detect ChatGPT generated content.

It also contains Infrastructure as Code (IaC) to create and manage AWS infrastructure for the task.  

## Project Structure
### Source code
The source code of this project is located in the folder:
```shell script
/src
```
There are three components in this folder. The "container" subfolder includes the source code for training, validation, serving, deoply and local testing. 

The "prepare_data" subfolder include scripts to generate training and validation data. We use webtext data as the human input data and label them as "real", as it's scraped all outbound links from Reddit. We use alpaca data as the ChatGPT generated content and label them as "fake", as alpaca data's "output" field is generated by ChatGPT. The links for the two dataset is as follows:      

https://paperswithcode.com/dataset/webtext

https://github.com/gururise/AlpacaDataCleaned

The "lambda_function" folder is a simple lambda function used in the Sagemaker IaC pipline to generate unique job ID for the SageMaker training job.


### Terraform configurations
The code for the Terraform part is in this repository in the folder:
```shell script
/terraform
```

It creates necessary resources and pipeline in AWS. Follow the steps below to deploy the infrastructure with Terraform.

```shell script

cd terraform/infrastructure

terraform init

terraform plan

terraform apply
```
Check the output and make sure the planned resources appear correctly and confirm with ‘yes’ in the apply stage if
everything is correct. Once successfully applied, you will get the URL for your ECR repository just created via Terraform.


## Build and Push your Docker Image to ECR


```shell script
cd src/container

docker build -t chat-gpt-detector .
```
After build the container, you can test the training, serving and endpoint locally by navigating to the "local_test" directory, please manually copy the training data and validation data from data directory to the src/container/local_test/test_dir/input directory.

```shell script
cd src/container/local_test

./train_local.sh chat-gpt-detector
```

You should be able to see the training started locally, after training finished, the model will be saved to local_test/test_dir/model. You can then start a local server by running the following.

```shell script

./serve_local.sh chat-gpt-detector

```
The server should be up and running. Now you can test the endpoint by running the following script. You can change the text in the payload. The response should return a json with label and probability.
```shell script
./predict.sh payload.json
```
After tested locally, you can push the docker image to ECR
```shell script
./build_and_push.sh chat-gpt-detector
```

## Run the ML pipeline

In order to train and run the ML pipeline in Sagemaker, go to Step Functions in AWS and start the execution. You can check progress of
SageMaker also in the Training Jobs section of SageMaker. Once the SageMaker Endpoint is created you can 
also check your SageMaker Endpoint. After running the State Machine in Step Functions successfully, you will see the
SageMaker Endpoint being created in the AWS Console in the SageMaker Endpoints section. Make sure to 
wait for the Status to change to “InService”.

## Test Sagemaker endpoint

Use the following script to test the Sagemaker endpoint.Please replace the endpoint with the actual endpoint.
```shell script
python src/container/local_test/test_enpoint.py
```

