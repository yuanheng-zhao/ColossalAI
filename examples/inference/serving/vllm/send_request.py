import requests

prompt = "Introduce some landmarks in Beijing"
input_data = {"prompt": prompt, "stream": False, "max_tokens": 64}
output = requests.post("http://localhost:8000/", json=input_data)

print(type(output))
print(output)
print(output.text)
