"""
GhostCoT Streaming Demo

Demonstrates how to use GhostCoT to enable Chain-of-Thought reasoning
for non-thinking models like GPT-4o, Claude, or DeepSeek-Chat.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python demo_ghostcot_stream.py
"""

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