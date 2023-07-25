# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import flask
import torch


# from transformers import *
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

id2label = {0: "fake", 1: "real"}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class DetectorService(object):
    model = None  # Where we keep the model when it's loaded
    tokenizer = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            model_name = "gpt2"
            cls.model = GPT2ForSequenceClassification.from_pretrained(model_name).to(
                device
            )
            cls.model.config.pad_token_id = cls.model.config.eos_token_id
            print(f"Loading saved checkpoint from {model_path}")
            data = torch.load(
                os.path.join(model_path, "best-model.pt"), map_location="cpu"
            )
            cls.model.load_state_dict(data["model_state_dict"])
            cls.model.eval()

        return cls.model

    @classmethod
    def get_tokenizer(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.tokenizer == None:
            model_name = "gpt2"
            cls.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name, add_special_tokens=True
            )
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

        return cls.tokenizer

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a string text): The text on which to do the classification"""
        model = cls.get_model()
        tokenizer = cls.get_tokenizer()

        tokens = tokenizer.encode(input)

        # Restrict the max sequence length to prevent potential error, the max length for this model is 1024, leave space for bos and eos
        max_token_length = 1022
        if len(tokens) > max_token_length:
            tokens = tokens[:max_token_length]
        tokens = torch.tensor(
            [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        ).unsqueeze(0)
        mask = torch.ones_like(tokens)

        with torch.no_grad():
            logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
            probs = logits.softmax(dim=-1).cpu().numpy()[0].tolist()
            predicted_class_id = logits.argmax().item()

        return predicted_class_id, probs


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        DetectorService.get_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def detect():
    """Do an inference on a single text input, predict it's class and also return the probility."""

    if flask.request.content_type == "application/json":
        content = flask.request.get_json()
        input_text = content["text"]
        predicted_class_id, probs = DetectorService.predict(input=input_text)

        return flask.jsonify(
            {
                "label": id2label[predicted_class_id],
                "probility": probs[predicted_class_id],
            }
        )
    else:
        return flask.Response(
            response="This predictor only supports json data",
            status=415,
            mimetype="text/plain",
        )
