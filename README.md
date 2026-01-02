# GhostCoT

A decorator to enable Chain-of-Thought (CoT) reasoning for non-thinking LLMs (e.g. gpt-4o-mini and deepseek-chat).

## Installation

```
pip install ghostcot
```

## Enable Non-Thinking LLMs to Think

**Question:** 2 + π ≈ ?

Using `ghostcot`, you can enable non-thinking LLMs (e.g. gpt-4o-mini) to reason and get:

> **Thinking:** `To estimate 2 + π, we need to know the approximate value of π. The value of π is approximately 3.14. Therefore: 2 + 3.14 = 5.14`

**✨ Final Answer:** **5.14**

You can run a demo with the following code:
```python
import ghostcot
api_key = "your-api-key" # OpenAI/DeepSeek/Other
base_url = None # None for OpenAI, otherwise for other LLMs
ghostcot.run_demo_cot_stream("What is 2 + π?", model_name="gpt-4o-mini", api_key=api_key, base_url=base_url)
```

![Demo](demo/demo_image.gif)

## How to Use 

The only change you need to make is to decorate your chat function of OpenAI API with `@ghostcot.enable_cot()`, and all the remaining code is the same as before.
```python
# Apply GhostCoT decorator
@ghostcot.enable_cot()
def chat(messages, **kwargs):
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )
```

To distinguish between reasoning content and final answer:
- **Non-streaming:** `response.choices[0].message.reasoning_content` and `response.choices[0].message.content`
- **Streaming:** `chunk.choices[0].delta.reasoning_content` and `chunk.choices[0].delta.content`

**Demos:**
- [Non-streaming demo](demo/demo_ghostcot_common.py)
- [Streaming demo](demo/demo_ghostcot_stream.py)








