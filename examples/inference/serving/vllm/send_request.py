import requests
from vllm import SamplingParams

# @ray.remote
# def send_query(text):
#     resp = requests.get("http://localhost:8000/?text={}".format(text))
#     return resp.text


# test_sentence = "Introduce some landmarks in Beijing"

# result = ray.get(send_query.remote(test_sentence))
# print("Result returned:")
# print(result)


# Refer to
# https://github.com/vllm-project/vllm/blob/55fe8a81ec5deab319c7c6b02913c21273b764ca/examples/llm_engine_example.py
test_prompts = [
    ("A robot may not injure a human being", SamplingParams(temperature=0.0)),
    ("To be or not to be,", SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
    (
        "What is the meaning of life?",
        SamplingParams(n=2, best_of=5, temperature=0.8, top_p=0.95, frequency_penalty=0.1),
    ),
    (
        "It is only with the heart that one can see rightly",
        SamplingParams(n=3, best_of=3, use_beam_search=True, temperature=0.0),
    ),
]

prompt = "Introduce some landmarks in Beijing"
input_data = {"prompt": prompt, "stream": False, "max_tokens": 64}
output = requests.post("http://localhost:8000/", json=input_data)

print(output)
# for line in output.iter_lines():
#     print(line.decode("utf-8"))
