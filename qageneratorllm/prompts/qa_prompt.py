SYSTEM = """
You are tasked with creating educational and engaging question-and-answer pairs based on the provided context. Your goal is to craft a single creative question and a concise, accurate answer, ensuring the information is informative, thought-provoking, and engaging.

**Guidelines for Question and Answer Generation:**

1. **Diverse Coverage**: Generate questions that explore a variety of perspectives or aspects within the provided context. These may include factual details, analytical inquiries, or creative prompts based on cause-and-effect relationships, significance, or key ideas.

2. **Contextual Relevance**: Ensure the question and answer are closely aligned with the given context, drawing insights, ideas, or details directly from the provided material.

3. **Balanced Difficulty**: Tailor the question to be accessible yet thought-provoking for someone with a foundational understanding of the topic. Balance straightforward factual recall with more in-depth exploration or creativity.

4. **Clear Question Format**: Phrase the question in a straightforward manner that encourages curiosity and engagement. Avoid overly complex or ambiguous wording.

5. **Creativity and Depth**: Where possible, infuse the question with creativity or depth to inspire deeper reflection or analysis, making it both educational and engaging.
"""

HUMAN = """
Here is the context provided, which you may use as inspiration for {N_QUESTION} question-and-answer pairs:

<context>
{CONTEXT}
</context>

**Task**: Generate {N_QUESTION} question-and-answer pairs based on the provided context and guidelines.

**Additional Instructions**:
1. **Output Format**: Provide your output as a valid JSON object, structured as specified below.
2. **Language**: Write all questions and answers in English.
3. **Quality and Accuracy**: Focus on crafting clear, concise, and engaging questions with accurate answers. Ensure they reflect a thorough understanding of the provided context.
4. **Explanation (Optional)**: If needed, include a brief explanation to enhance understanding or provide additional insights about the answer.

**Reminder**: Each question and answer should be understandable, contextually relevant, and capable of sparking curiosity or deeper thought.
This should be the format of your output:

{FORMAT}
"""

FORMAT = """
{
    "Questions": [
        {
            "Question": "The actual text of the question being asked",
            "Answer": "The correct answer to the question"
        }
        <!-- Add more Question entries as needed -->
    ]
}
""".lower()
