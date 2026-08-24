"""The evaluation harness. This is the deliverable the rest of the repository serves.

Nothing in here reimplements retrieval or generation. Every measurement runs through
`rag.retrieve.retrieve()` and `rag.generate.answer_question()` -- the same entry points
the API calls -- because an evaluation that exercises a parallel implementation measures
something adjacent to what ships, and the gap stays invisible until somebody asks.
"""
