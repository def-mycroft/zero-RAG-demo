# The Why and What of Retrieval Augmented Generation

Large language models like ChatGPT can process large quantities of text practically instantly. However, there are still limits to how much information can be given to ChatGPT with the context of a conversation. As of this writing, ChatGPT's flagship GPT-4o model can receive a maximum of about 125,000 words. 

This quantity of words, 125,000, is roughly equivalent to that of a book. This is ample capacity for many use cases, but what if you want to use ChatGPT to analyze the text equivalent of *an entire bookshelf* of books? 

We can't feed the text equivalent of an entire bookshelf of books to ChatGPT because of the aforementioned technical limitation, but we can *select relevant subsets of the text and give these subsets to ChatGPT*. This describes Retrieval Augmented Generation (RAG). 

The purpose of this document is to demonstrate a use case for RAG and to do so in a manner which abstracts from specific technical details. I've authored code which fully implements RAG on a dataset of corporate quarterly earnings reports and provided a Jupyter notebook that describes the business problem that is being addressed and describes how RAG is being implemented to provide a solution. 

[Here is the Jupyter notebook that presents this project](notebooks/Conceptual%20Introduction%20to%20RAG.ipynb)

