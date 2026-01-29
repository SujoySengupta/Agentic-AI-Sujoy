# 5 Levels of Text Splitting: A Technical Guide

## Overview

This documentation covers five distinct strategies for splitting (or "chunking") text to optimize the performance of Large Language Model (LLM) applications. The goal is to fit data into context windows while maintaining semantic meaning.

### Prerequisites

The workflow relies on the `langchain` ecosystem. You must install the following libraries before running the examples:

```bash
pip install langchain langchain-text-splitters langchain_experimental langchain_openai openai

```

---

## Level 1: Character Splitting

**Concept:** This is the most basic form of chunking. It divides text into -character sized chunks regardless of content, structure, or semantic meaning.

* **Pros:** Easy and simple to implement.
* **Cons:** Very rigid; frequently breaks contexts (e.g., splitting a word in half).

**Key Parameters:**

* `chunk_size`: The number of characters per chunk (e.g., 35).
* `chunk_overlap`: The number of characters to overlap between sequential chunks to preserve some context.

**Usage:**

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=35, 
    chunk_overlap=4, 
    separator='', 
    strip_whitespace=False
)
docs = text_splitter.create_documents([text])

```

---

## Level 2: Recursive Character Text Splitting

**Concept:** This method respects the structure of the document. It attempts to split text using a hierarchical list of separators (e.g., Paragraphs  New lines  Spaces) to keep related text together.

**Default Separator Hierarchy:**

1. `"\n\n"` (Paragraph breaks)
2. `"\n"` (New lines)
3. `" "` (Spaces)
4. `""` (Characters)

**Usage:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=65, 
    chunk_overlap=0
)
docs = text_splitter.create_documents([text])

```

*Note: This is generally the recommended starting point for most prose text applications.*

---

## Level 3: Document Specific Splitting

**Concept:** This level utilizes splitters optimized for specific data formats (Markdown, Python, JavaScript, etc.) to ensure syntax and structural hierarchy are preserved.

### 1. Markdown

Splits based on headers (`#`, `##`, `###`) to keep sections together.

```python
from langchain_text_splitters import MarkdownTextSplitter

splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)
docs = splitter.create_documents([markdown_text])

```

### 2. Python Code

Splits based on class definitions and function definitions to ensure code remains executable or logically grouped.

```python
from langchain_text_splitters import PythonCodeTextSplitter

python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)
docs = python_splitter.create_documents([python_code])

```

### 3. JavaScript

Uses separators like `function`, `const`, `let`, `if`, `for`, etc.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, 
    chunk_size=65, 
    chunk_overlap=0
)

```

---

## Level 4: Semantic Splitting

**Concept:** Instead of splitting by characters, this method groups text based on semantic meaning. It uses embeddings to determine the cosine similarity between sentences. If the similarity between two sentences drops below a specific threshold (representing a topic change), a split occurs.

**Workflow:**

1. Sentences are embedded using a model (e.g., OpenAI).
2. Cosine similarity is calculated between sequential sentences.
3. Breaks are created at "dips" in similarity.

**Usage:**

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)
docs = semantic_splitter.split_text(text)

```

---

## Level 5: Agentic Splitting

**Concept:** An experimental and advanced method that uses an LLM (acting as an agent) to read the text and determine logical split points based on human-like understanding of the content.

**Workflow:**

1. **Prompting:** The LLM is prompted to act as a "chunking expert."
2. **Marking:** The LLM inserts a unique token (e.g., `<<<SPLIT>>>`) where it identifies a natural topic boundary.
3. **Splitting:** The system programmatically splits the text string at those tokens.

**Usage:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.
Put "<<<SPLIT>>>" between chunks.
Text: {text}
"""
response = llm.invoke(prompt)
chunks = response.content.split("<<<SPLIT>>>")

```

---

## Evaluations

The notebook emphasizes that chunking strategies must be tested using retrieval evaluations (Evals). Recommended frameworks include:

* **LangChain Evals**
* **Llama Index Evals**
* **RAGAS Evals**