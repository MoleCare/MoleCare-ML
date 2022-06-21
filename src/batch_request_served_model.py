import requests

response = requests.post(
    "http://localhost:8501/v1/models/classifier:predict",
    json={
        "signature_name": "serving_default",  # can be omitted
        "inputs": {
            "image_bytes": [image.numpy().decode("utf-8")
                            for image in image_bytes][:2],  # batch request
        },
    },
)
