system = """
You are tasked with creating multiple-choice questions (MCQs) about African history. These questions should be accessible to anyone with a foundational knowledge of African history, while primarily relying on the provided context. Your goal is to craft engaging, educational questions that assess a broad understanding of African history.

**Guidelines for Question Generation:**

1. **Diverse Coverage**: Create questions that span various facets of African history, including significant events, influential figures, cultures, and historical periods. Emphasize the historical period when explicitly mentioned in the context, as this can impact the accuracy of answers.
  
2. **Contextual Relevance**: Align questions closely with the given context, ensuring that they primarily draw from this information while remaining rooted in commonly understood historical themes.

3. **Balanced Difficulty**: Make the questions challenging yet fair, suitable for someone with a reasonable grasp of African history. Balance straightforward factual recall with questions exploring cause-and-effect relationships, historical significance, or major turning points.

4. **Varied Question Types**: Include a mix of question styles, such as recall-based questions, understanding of cause-effect relationships, and inquiries into the significance of events or figures. Avoid over-reliance on any one type to maintain variety and engagement.

5. **Period Emphasis**: When applicable, reference the historical period in which the event, figure, or culture was prominent. This can clarify potential dependencies on time-specific information, enhancing both historical accuracy and depth.
"""

human = """
Here is the context provided from the book "{SOURCE}", which you may use as inspiration for {N_QUESTION} questions:

<context source="{SOURCE}">
{CONTEXT}
</context>

**Task**: Generate {N_QUESTION} multiple-choice questions (MCQs) according to the guidelines.

**Additional Instructions**:
1. **Output Format**: Provide your output as a valid JSON object, structured as specified below.
2. **Language**: Write all questions and answer choices in English.
3. **Quality and Accuracy**: Focus on factual and concise questions. Ensure that the questions reflect a thorough understanding of the provided context and African history.
4. **Answer Options**: Include up to five answer choices (A, B, C, D, and E) for each question.
5. **Correct Answers**: Mark at least one correct answer per question.
6. **Question Wording**: Avoid phrasing like "According to the context" or "As mentioned in the passage"... All explanations should be factual and based on widely accepted knowledge of African history.

**Reminder**: Each question and explanation should be understandable and answerable based on general understanding of the context.
"""
