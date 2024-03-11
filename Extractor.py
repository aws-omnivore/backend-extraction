import boto3

import os
import io
import time
import base64
from PIL import Image

import torch
from Resnet import resnet18
from torchvision import transforms

import requests
from flask_cors import CORS
from flask import Flask, request, jsonify


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", required=True, default=True)
parser.add_argument("-p", "--service_prefix", required=False, default="/api/v1/extracor/")

args = vars(parser.parse_args())



s3_client = boto3.client('s3')


MODEL_BUCKET_NAME = "extract-signboard-models"
IMAGE_BUCKET_NAME='restaurants-image'




store_names = ["찐퍼", "찐퍼", "찐퍼", "찐퍼"]
def calc_landmark(La, Lo):
    pass

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])



if args['test']:
    
    print("Test Environment")
    LATEST_MODEL_NAME = "0030_resnet18_trained_0.003.pth"
    landmark = "q"
    
    num_classes=4
    print("Model Self Test")
    model = resnet18(num_classes=num_classes).cpu()
    model.load_state_dict(torch.load(LATEST_MODEL_NAME, map_location=torch.device('cpu')))
    model.eval()

    img = Image.open('testimg.JPG')
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    img = transform(img).unsqueeze(0)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

    print("Test Predicted", predicted, store_names[predicted.item()])

    
else:
    landmark = calc_landmark(LA, LO)
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=MODEL_BUCKET_NAME, Prefix=landmark)

    model_names = []
    for page in page_iterator:
        model_names += [keys['Key'] for keys in page['Contents']]
    
    LATEST_MODEL_NAME = sorted(model_names)[-1]
    
    if not os.path.isfile('/tmp/'+LATEST_MODEL_NAME):
        s3_client.download_file(MODEL_BUCKET_NAME, LATEST_MODEL_NAME, '/tmp/'+LATEST_MODEL_NAME)
    LATEST_MODEL_NAME = '/tmp/'+LATEST_MODEL_NAME
    num_classes=4


if torch.cuda.is_available():
    model = resnet18(num_classes=num_classes).cuda()
    model.load_state_dict(torch.load(LATEST_MODEL_NAME))
else:
    model = resnet18(num_classes=num_classes).cpu()
    model.load_state_dict(torch.load(LATEST_MODEL_NAME, map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/extract', methods=['POST','GET'])
def extractor():
    try:
        la, lo = request.headers.get('La') , request.headers.get('Lo')
        language=request.headers.get('Language')
        print("la, lo, language", la, lo, language)
        image_file = request.files['image']
        image_bytes = io.BytesIO(image_file.read())
        img = Image.open(image_bytes)
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = transform(img).unsqueeze(0)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        store_names = ["찐퍼", "찐퍼", "찐퍼", "찐퍼"]
        target_url = "http://translate-service.fs-translate.svc.cluster.local/api/v1/translate"
        # "http://translate-service.fs-translate.svc.cluster.local/api/v1/translate/"
        store_name = store_names[predicted.item()]
        # target_url = target_url+"?name="+store
        target_url = target_url + "?name=" + store_name
        # target_url = target_url+"?name‎="+store_names[predicted.item()]
        
        # target_url += store_name  # 경로 수정
        
        
        
        # translate_response = requests.post(target_url, json={'name': predicted.item(), 'Language' : language})
        translate_response = requests.get(target_url, headers={'Language': language})
        
        return jsonify({'log': f'target_url, {target_url}, {translate_response}'}), 500
        if translate_response.status_code == 200:
            translate_data = translate_response.json()
            return jsonify(translate_data)
            print("Result sent successfully")
        else:
            return jsonify({'error': f'Translate service error, {translate_response}'}), 500
            print("Failed to send result")
        
    except Exception as e:
        print(e)
        return jsonify({'error':str(e)}), 500

if __name__ == '__main__':
    print("IN here")
    app.run(host='0.0.0.0', port=5000)
