from setuptools import setup, find_packages

description = "A decorator to enable Chain-of-Thought (CoT) reasoning for non-thinking LLMs."

setup(
    name="ghostcot",
    version="0.0.4",
    author="Jun Hu",
    author_email="hujunxianligong@gmail.com",
    description=description,
    long_description=description,
    long_description_content_type="text/plain",
    url="https://github.com/CrawlScript/GhostCoT",
    packages=find_packages(
        exclude=[
            "demo",
            "data",
            "dist",
            "doc",
            "docs",
            "logs",
            "models",
            "test"
            ]),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
    ],
)