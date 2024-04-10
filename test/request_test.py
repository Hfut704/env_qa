import requests
import json


def send_json_request(url, json_data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(json_data), headers=headers)

    if response.status_code == 200:
        print("Request was successful")
        return response.json()
    else:
        print("Request failed with status code:", response.status_code)
        return None

# 示例 JSON 数据
json_data = {
    "question": "你好"
}

# 发送请求的 URL
url = "http://114.213.210.46:5500/v1/query2kb"

# 调用函数发送请求
#
res = send_json_request(url, json_data)
print(res)