You are helping build an evaluation set for a retrieval system over machine learning
papers. You will be shown one passage from one paper. Write a single question that the
passage answers, and the answer.

Requirements for the question:

1. It must be answerable from the passage alone, and the passage must contain the whole
   answer, not a hint towards it.
2. It must be **standalone**. A reader who has never seen the passage must be able to
   understand what is being asked. Never write "according to the passage", "in this
   work", "the authors" without naming what they did, or "this method" without naming it.
3. It must not quote the passage. If someone searched the corpus using your question
   verbatim, the passage should be findable because it is *about* the same thing, not
   because it shares a rare string.
4. Prefer specifics over generalities: a number, a name, a mechanism, a stated limitation.
   Avoid questions whose answer is a topic rather than a fact.
5. It must be a question a person might actually ask about this literature.

Requirements for the answer:

- One or two sentences. State the fact, nothing else.
- Use only what the passage says.

Respond as JSON and nothing else:

{"question": "...", "expected_answer": "...", "difficulty": "easy|medium|hard", "type": "factual|numeric"}

Use `numeric` when the answer is a figure, a measurement or a table value. Use `hard` only
when answering requires combining several statements within the passage.
