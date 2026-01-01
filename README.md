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
api_key = "your-api-key"
ghostcot.run_demo_cot_stream("What is 2 + π?", model_name="gpt-4o-mini", api_key=api_key)
```


## How to Use

The only change you need to make is to decorate your chat function of OpenAI API with `@enable_cot()`. Then, the output contains both reasoning content and final answer, and you can use `chunk.choices[0].delta.reasoning_content` and `chunk.choices[0].delta.content` to distinguish between them. See the demo for more details.



```python
import os
from ghostcot import enable_cot
from openai import OpenAI

model_name = "gpt-4o-mini"

def main():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # For DeepSeek, use their base URL
    client = OpenAI(
        api_key=api_key,
    )
    
    # Apply GhostCoT decorator
    @enable_cot()
    def chat(messages, **kwargs):
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs
        )
    
    # Test questions
    questions = [
        "2 + π ≈ ?",
        # "Calculate (5 + x) * 2 where x = 10"
    ]
    
    for i, question in enumerate(questions, 0):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}: {question}")
        print('='*60)
        print("\n👻 Ghost Thinking (Reasoning):")
        print('-'*60)
        
        reasoning_done = False
        
        for chunk in chat(
            messages=[{"role": "user", "content": question}],
            stream=True
        ):
            # Print reasoning (thinking process)
            if chunk.choices[0].delta.reasoning_content:
                print(chunk.choices[0].delta.reasoning_content, end='', flush=True)
            
            # Print final answer
            if chunk.choices[0].delta.content:
                if not reasoning_done:
                    print("\n" + '-'*60)
                    print("✨ Final Answer:")
                    print('-'*60)
                    reasoning_done = True
                print(chunk.choices[0].delta.content, end='', flush=True)
        
        print("\n")


if __name__ == "__main__":
    main()
```

Original output:
```
============================================================
Question 0/1: 2 + π ≈ ?
============================================================

👻 Ghost Thinking (Reasoning):
------------------------------------------------------------

To estimate 2 + π, we need to know the approximate value of π. The value of π is approximately 3.14. Therefore: 2 + 3.14 = 5.14

------------------------------------------------------------
✨ Final Answer:
------------------------------------------------------------

5.14
```







