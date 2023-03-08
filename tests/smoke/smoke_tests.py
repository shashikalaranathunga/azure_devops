import requests
import json

req_sample = {"SepalLengthCm": [6.6], "SepalWidthCm": [3], "PetalLengthCm": [4.4], "PetalWidthCm": [1.4]}

def test_ml_service(scoreurl):
    assert scoreurl != None
    headers = {'Content-Type':'application/json'}
    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    assert resp.status_code == requests.codes["ok"]
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0

def test_prediction(scoreurl):
    assert scoreurl != None
    headers = {'Content-Type':'application/json'}
    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    print(resp.text)
    resp_json = json.loads(resp.text)[0]
    assert (resp_json == 1) or (resp_json == 2) or (resp_json == 0)