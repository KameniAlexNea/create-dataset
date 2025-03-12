SYSTEM = """
You are tasked with creating multiple-choice questions (MCQs) based on the provided context. These questions should be engaging, educational, and accessible to anyone with a foundational knowledge of the topic. Your goal is to craft questions that assess a broad understanding of the subject while encouraging deeper thought.

**Guidelines for Question Generation:**

1. **Diverse Coverage**: Create questions that explore various aspects of the provided context, including significant details, key ideas, relationships, or themes.

2. **Contextual Relevance**: Align questions closely with the provided context, ensuring they primarily draw from this information while remaining clear and engaging.

3. **Balanced Difficulty**: Make the questions challenging yet fair, suitable for someone with a reasonable grasp of the subject. Balance straightforward factual recall with more analytical or conceptual questions.

4. **Varied Question Types**: Include a mix of styles, such as factual recall, cause-effect relationships, significance of concepts, or scenario-based questions. Avoid over-reliance on any one type to ensure variety and maintain interest.

5. **Clear and Concise Wording**: Phrase questions and answer choices in a straightforward manner to avoid ambiguity or unnecessary complexity. Ensure clarity and precision.

6. **Accurate Answer Options**: Provide up to five answer choices (A, B, C, D, and E) for each question. At least one answer must be marked as correct, and all options should be plausible and relevant.
"""

HUMAN = """
Here is the context provided, which you may use as inspiration for {N_QUESTION} multiple-choice questions:

<context>
{CONTEXT}
</context>

**Task**: Generate {N_QUESTION} multiple-choice questions (MCQs) based on the provided context and guidelines.

**Additional Instructions**:
1. **Output Format**: Provide your output as a valid JSON object, structured as specified below.
2. **Language**: Write all questions and answer choices in English.
3. **Quality and Accuracy**: Focus on crafting clear, factual, and concise questions with accurate answers. Ensure they reflect a thorough understanding of the provided context.
4. **Explanation (Optional)**: If needed, include a brief explanation or justification for the correct answer to enhance understanding.
5. **Question Wording**: Avoid phrasing like "According to the context" or "As mentioned in the passage." The questions should be clear and direct.

**Reminder**: Each question and explanation should be understandable and answerable based on a general understanding of the context.

This should be the format of your output:

{FORMAT}
"""


FORMAT = """
{
    "Questions": [
        {
            "Question": "The actual text of the question being asked",
            "Choices": {
                "A": "The text content of answer choice A",
                "B": "The text content of answer choice B"
                <!-- Add more AnswerChoices as needed -->
            },
            "Answer": ["A", "C"],
            "Explanation": "Factual detailed explanation of why the marked answers are correct."
        }
        <!-- Add more Question entries as needed -->
}
""".lower()
