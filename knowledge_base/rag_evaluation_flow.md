# RAG Evaluation Flow

Use retrieval before evaluation so the judge is grounded by stable context.

Evaluation flow:

1. receive user prompt
2. generate candidate responses
3. retrieve rubric and evaluation guidance from the knowledge base
4. judge each response with the retrieved context
5. return scores and explanations
6. gate on threshold when needed

RAG grounding reduces unsupported scoring and makes the evaluator more inspectable.
