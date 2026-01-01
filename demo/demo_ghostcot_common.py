"""
GhostCoT Common Demo

Demonstrates how to use GhostCoT to enable Chain-of-Thought reasoning
for non-thinking models like GPT-4o, Claude, or DeepSeek-Chat.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python demo_ghostcot_common.py
"""

import os
import ghostcot
from openai import OpenAI

model_name = "gpt-4o-mini"

def main():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
    )
    
    # Apply GhostCoT decorator
    @ghostcot.enable_cot()
    def chat(messages, **kwargs):
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs
        )
    
    # Test question
    question = "2 + π ≈ ?"
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    response = chat(
        messages=[{"role": "user", "content": question}],
        stream=False
    )
    
    print("\n👻 Ghost Thinking (Reasoning):")
    print('-'*60)
    print(response.choices[0].message.reasoning_content)
    
    print("\n" + '-'*60)
    print("✨ Final Answer:")
    print('-'*60)
    print(response.choices[0].message.content)
    print()


if __name__ == "__main__":
    main()