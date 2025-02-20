# TinyRag
RAG（Retrieval-Augmented Generation，检索增强生成），顾名思义，是通过检索的方法来增强模型的生成能力。
LLM在实际应用中暴露出一些问题。一方面，存在知识局限，具有时效性和覆盖范围的限制，难以掌握最新的信息以及所有专业领域的知识。另一方面，生成结果的可靠性欠佳，容易出现 “幻觉” 现象，即生成与事实不符的内容。

在这种情况下，RAG应运而生。简单而言，rag将信息检索与LLM生成结合。在面对用户的提问时，RAG 先通过检索模块从数据库中找出最相关、最准确且最新的信息片段，再将这些信息输入到生成模块。生成模块结合LLM输出高质量的回答。这一过程大大减少了错误和 “幻觉” 内容的产生，显著提升了回答的准确性和时效性。

## rag基本结构


