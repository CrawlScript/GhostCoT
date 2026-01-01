# GhostCoT

A decorator to enable Chain-of-Thought (CoT) reasoning for non-thinking LLMs (e.g. gpt-4o-mini and deepseek-chat).

## Installation

```
pip install ghostcot
```

## Usage

**Question 0/1:** 2 + π ≈ ?

Using `ghostcot` + gpt-4o-mini (which is not a thinking model) by runing:
```
python demo/demo_ghostcot_stream.py
```
we can get the following output:

> **Thinking:** `To estimate 2 + π, we need to know the approximate value of π. The value of π is approximately 3.14. Therefore: 2 + 3.14 = 5.14`

**✨ Final Answer:** **5.14**



## How to Use

Decorate your chat function with `@enable_cot()`:
```python
from ghostcot import enable_cot
from openai import OpenAI

api_key = "your-api-key"
client = OpenAI(api_key=api_key)

@enable_cot()
def chat(messages, **kwargs):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        **kwargs
    )

messages=[{"role": "user", "content": "2 + π ≈ ?"}]

# Stream the response
thinking_started = False
content_started = False

for chunk in chat(messages=messages, stream=True):
    # Print reasoning (thinking process)
    if chunk.choices[0].delta.reasoning_content:
        if not thinking_started:
            print("----Think----")
            thinking_started = True
        print(chunk.choices[0].delta.reasoning_content, end='')
    
    # Print final answer
    if chunk.choices[0].delta.content:
        if not content_started:
            print("\n----Content----")
            content_started = True
        print(chunk.choices[0].delta.content, end='')

print("")

```







