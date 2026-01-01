"""
GhostCoT - Elicit Chain-of-Thought reasoning from any LLM via prompt injection.

Usage:
    from ghostcot import enable_cot
    from openai import OpenAI
    
    client = OpenAI()
    
    @enable_cot()
    def chat(messages, **kwargs):
        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            **kwargs
        )
    
    for chunk in chat(messages=[{"role": "user", "content": "Question"}], stream=True):
        if chunk.choices[0].delta.reasoning_content:
            print(chunk.choices[0].delta.reasoning_content, end='')
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='')
"""

from functools import wraps

__version__ = "0.0.1"
__all__ = ["enable_cot"]


def generate_cot_instruction(start_tag, end_tag):
    """Generate CoT instruction prompt"""
    return f"""
CRITICAL: You MUST wrap your reasoning in {start_tag} tags.

Format:
{start_tag}
your reasoning
{end_tag}

your answer
"""


def inject_cot_instruction(messages, cot_instruction):
    """Inject CoT instruction into messages"""
    new_messages = []
    system_found = False
    
    for msg in messages:
        if msg.get('role') == 'system':
            new_messages.append({
                'role': 'system',
                'content': msg['content'] + '\n\n' + cot_instruction
            })
            system_found = True
        else:
            new_messages.append(msg)
    
    if not system_found:
        new_messages.insert(0, {
            'role': 'system',
            'content': cot_instruction
        })
    
    return new_messages


class CoTChunk:
    """Wrapper for streaming chunks with reasoning_content"""
    def __init__(self, original_chunk, reasoning_content=None, content=None):
        self.original = original_chunk
        self.choices = [CoTChoice(reasoning_content, content)]


class CoTChoice:
    def __init__(self, reasoning_content, content):
        self.delta = CoTDelta(reasoning_content, content)


class CoTDelta:
    def __init__(self, reasoning_content, content):
        self.reasoning_content = reasoning_content
        self.content = content


class CoTStreamWrapper:
    """Stream wrapper that parses reasoning tags"""
    def __init__(self, stream, start_tag, end_tag):
        self.stream = stream
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.buffer = ""
        self.state = "INIT"
        self.output_pos = 0
    
    def __iter__(self):
        for chunk in self.stream:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            
            self.buffer += chunk.choices[0].delta.content
            
            if self.state == "INIT" and self.start_tag in self.buffer:
                start_pos = self.buffer.find(self.start_tag)
                self.output_pos = start_pos + len(self.start_tag)
                self.state = "THINKING"
                continue
            
            if self.state == "THINKING" and self.end_tag in self.buffer:
                end_pos = self.buffer.find(self.end_tag)
                thinking_content = self.buffer[self.output_pos:end_pos]
                if thinking_content:
                    yield CoTChunk(chunk, reasoning_content=thinking_content, content=None)
                
                self.output_pos = end_pos + len(self.end_tag)
                self.state = "ANSWER"
                continue
            
            if self.state == "THINKING":
                safe_end = len(self.buffer) - len(self.end_tag)
                if safe_end > self.output_pos:
                    output = self.buffer[self.output_pos:safe_end]
                    if output:
                        yield CoTChunk(chunk, reasoning_content=output, content=None)
                    self.output_pos = safe_end
            
            elif self.state == "ANSWER":
                if len(self.buffer) > self.output_pos:
                    output = self.buffer[self.output_pos:]
                    if output:
                        yield CoTChunk(chunk, reasoning_content=None, content=output)
                    self.output_pos = len(self.buffer)
            
            elif self.state == "INIT":
                safe_end = len(self.buffer) - len(self.start_tag)
                if safe_end > self.output_pos:
                    output = self.buffer[self.output_pos:safe_end]
                    if output:
                        yield CoTChunk(chunk, reasoning_content=None, content=output)
                    self.output_pos = safe_end
        
        if self.output_pos < len(self.buffer):
            remaining = self.buffer[self.output_pos:]
            if self.state == "THINKING":
                yield CoTChunk(None, reasoning_content=remaining, content=None)
            else:
                yield CoTChunk(None, reasoning_content=None, content=remaining)


def enable_cot(start_tag="<thinking>", end_tag="</thinking>", instruction=None):
    """
    Decorator to enable Chain-of-Thought reasoning for any LLM.
    
    Args:
        start_tag: Tag marking the start of reasoning (default: "<thinking>")
        end_tag: Tag marking the end of reasoning (default: "</thinking>")
        instruction: Custom CoT instruction (default: auto-generated)
    
    Returns:
        Decorated function that returns CoTStreamWrapper
    
    Example:
        @enable_cot()
        def chat(messages, **kwargs):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                **kwargs
            )
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'messages' in kwargs:
                cot_inst = instruction or generate_cot_instruction(start_tag, end_tag)
                kwargs['messages'] = inject_cot_instruction(kwargs['messages'], cot_inst)
            
            original_stream = func(*args, **kwargs)
            return CoTStreamWrapper(original_stream, start_tag, end_tag)
        
        return wrapper
    return decorator




def run_demo_cot_stream(question, model_name="gpt-4o-mini", api_key=None, base_url=None):
    """
    Quick demo function to run Chain-of-Thought reasoning.
    
    Args:
        question: The question to ask
        model_name: Model name (default: "gpt-4o-mini")
        api_key: API key (default: from OPENAI_API_KEY env var)
        base_url: Optional base URL for custom endpoints (e.g., DeepSeek)
    
    Example:
        >>> from ghostcot import demo_cot_stream
        >>> demo_cot_stream("What is 2 + π?")
    """
    import os
    from openai import OpenAI
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please provide api_key or set OPENAI_API_KEY environment variable")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    @enable_cot()
    def chat(messages, **kwargs):
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs
        )
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
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