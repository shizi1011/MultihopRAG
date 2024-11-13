query_template = """Write a simple search query that will help answer a complex question.
---

Follow the following format.

Context: may contain relevant facts

Question: ${{question}}

Reasoning: Let's think step by step in order to ${{produce the query}}. We ...

Query: ${{query}}

---

Context : {context}

Question : {question}

Reasoning : Let's think step by step in order to"""

answer_decision_template = """Is the retrieved context enough to answer the question ?

---

Follow the following format.

Context: may contain relevant facts

Question: ${{question}}

Reasoning: Let's think step by step in order to ${{produce the decision}}. We ...

Decision: only contain Yes or No

---
Context : {context}

Question : {question}

Reasoning : Let's think step by step in order to"""

answer_template = """Answer questions with short factoid answers

---

Follow the following format.

Context: may contain relevant facts

Question: ${{question}}

Reasoning: Let's think step by step in order to ${{produce the answer}}. We ...

Answer: often between 1 and 5 words

---

Context : {context}

Question : {question}

Reasoning : Let's think step by step in order to"""