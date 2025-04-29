import requests

url = 'http://127.0.0.1:5000/predict'
image_path = 'main/test/test_img1.jpg'

with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response Text:", response.text)

# Now try to decode JSON if response is not empty
try:
    print("JSON:", response.json())
except requests.exceptions.JSONDecodeError:
    print("❌ Failed to parse JSON. Maybe server returned an error page or empty response.")
