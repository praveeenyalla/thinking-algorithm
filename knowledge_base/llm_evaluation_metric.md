# LLM Evaluation Metric

An evaluation test case should include the problem, candidate answer, and retrieval context.

The judge should produce:

- a numeric score
- an optional reason
- a threshold decision

Core rubric dimensions:

- correctness
- clarity
- usefulness

The metric should support CI-style pass or fail decisions for a generated answer.
