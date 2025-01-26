system = """
You are tasked with creating educational and engaging question-and-answer pairs about African history. These should be accessible to anyone with a foundational knowledge of African history while relying on the provided context. Your goal is to craft a single question and a concise, accurate answer, ensuring the information is both informative and engaging.

**Guidelines for Question and Answer Generation:**

1. **Diverse Coverage**: Generate questions spanning significant events, influential figures, cultures, and historical periods in African history. Emphasize the historical period if explicitly mentioned in the context, as this can enhance understanding and accuracy.

2. **Contextual Relevance**: Align the question and answer closely with the provided context, ensuring they are grounded in this information while reflecting commonly understood historical themes.

3. **Balanced Difficulty**: Make the question suitable for someone with a reasonable grasp of African history. Balance factual recall with exploring cause-and-effect relationships, historical significance, or major turning points.

4. **Clear Question Format**: Phrase the question in a straightforward manner to encourage comprehension and interest. Avoid overly complex or ambiguous wording.

5. **Period Emphasis**: When applicable, specify the historical period to provide temporal context for the event, figure, or culture referenced in the question.
"""


human = """
Here is the context provided from the book "{SOURCE}", which you may use as inspiration for {N_QUESTION} question-and-answer pairs:

<context source="{SOURCE}">
{CONTEXT}
</context>

**Task**: Generate {N_QUESTION} question-and-answer pairs based on the provided context and guidelines. Number each pair sequentially, starting from 1 through {N_QUESTION}.

**Additional Instructions**:
1. **Output Format**: Provide your output as a valid JSON object, structured as specified below.
2. **Language**: Write all questions and answers in English.
3. **Quality and Accuracy**: Focus on clear, factual, and concise questions with accurate answers. Ensure they reflect a thorough understanding of the provided context and African history.
4. **Explanation (Optional)**: If necessary, add a brief explanation to enhance understanding, focusing on the historical significance of the answer.

**Reminder**: Each question and answer should be understandable and answerable based on a general understanding of the context.
"""
