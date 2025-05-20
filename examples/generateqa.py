from typing import Dict, List, Tuple

import gradio as gr
from llama_index.core.schema import TextNode

from qageneratorllm.generator import LLMProviderType, QuestionGenerator, QuestionType
from qageneratorllm.loader import chunk_document, sort_chunked_documents
from qageneratorllm.qa_dataclass import OutputType


def process_document(file_path) -> Tuple[List[Dict], List[TextNode]]:
    """Process uploaded document and return chunks with metadata."""
    if file_path is None:
        return [], []

    # Chunk the document and organize by relationships
    chunked_doc = chunk_document(file_path)
    organized_chunks = sort_chunked_documents(chunked_doc)

    # Flatten the nodes but keep relationship info
    chunked_nodes = []
    node_map = {}  # Map node_id to index for quick lookup

    # First pass: collect all nodes
    for node_id, nodes in organized_chunks.items():
        if nodes:  # Make sure we have at least one node
            for node in nodes:
                idx = len(chunked_nodes)
                chunked_nodes.append(node)
                node_map[node.node_id] = idx

    # Prepare chunks for display with hierarchy info
    chunk_data = []
    for i, node in enumerate(chunked_nodes):
        # Get header level directly from node metadata
        header_level = node.metadata.get("level")
        header_tag = f"h{header_level}" if header_level else "p"

        # Get parent information from relationships if available
        parent_id = None
        if (
            node.relationships and "4" in node.relationships
        ):  # "4" typically indicates parent
            parent_relation = node.relationships["4"]
            if isinstance(parent_relation, dict):
                parent_id = parent_relation.get("node_id")

        # Get children from relationships if available
        children_ids = []
        if (
            node.relationships and "5" in node.relationships
        ):  # "5" typically indicates children
            children_relation = node.relationships["5"]
            if isinstance(children_relation, list):
                children_ids = [
                    child.get("node_id")
                    for child in children_relation
                    if isinstance(child, dict)
                ]

        # Create context path from metadata if available
        context_path = node.metadata.get("context", "")

        chunk_data.append(
            {
                "id": i,
                "node_id": node.node_id,
                "text": node.text,
                "header_level": header_level,
                "header_tag": header_tag,
                "parent_id": parent_id,
                "children_ids": children_ids,
                "context_path": context_path,
                "metadata": node.metadata,
            }
        )

    return chunk_data, chunked_nodes


def filter_chunks_by_header(chunks: List[Dict], header_level: str) -> List[Dict]:
    """Filter chunks by header level with option to include children."""
    if header_level == "all":
        return chunks

    level = int(header_level) if header_level.isdigit() else None

    # First identify chunks that match the specified level
    matched_chunks = []
    if level is None:
        matched_chunks = [chunk for chunk in chunks if chunk["header_level"] is None]
    else:
        matched_chunks = [chunk for chunk in chunks if chunk["header_level"] == level]

    # Include chunks that are children of matched headers
    matched_ids = {chunk["node_id"] for chunk in matched_chunks}
    included_chunks = matched_chunks.copy()

    # Also include any children of the matched headers
    for chunk in chunks:
        if chunk["parent_id"] in matched_ids and chunk not in included_chunks:
            included_chunks.append(chunk)

    return included_chunks


def format_chunk_for_display(chunk: Dict) -> str:
    """Format a chunk for display in the UI with context information."""
    header_tag = chunk.get("header_tag", "p")
    text = chunk.get("text", "")
    context_path = chunk.get("context_path", "")

    # Format with context path if available
    if context_path:
        context_display = f"<div class='context-path'>{context_path}</div>"
    else:
        context_display = ""

    if header_tag.startswith("h"):
        return f"{context_display}<{header_tag}>{text}</{header_tag}>"
    return f"{context_display}<p>{text}</p>"


def generate_questions(
    chunks: List[Dict],
    selected_chunks: List[int],
    question_type: str,
    provider: str,
    num_questions: int,
) -> str:
    """Generate questions for selected chunks."""
    if not selected_chunks:
        return "Please select at least one chunk to generate questions."

    # Get the text from selected chunks
    selected_texts = []
    for chunk_id in selected_chunks:
        for chunk in chunks:
            if chunk["id"] == chunk_id:
                selected_texts.append(chunk["text"])
                break

    combined_text = "\n\n".join(selected_texts)

    # Initialize question generator
    question_generator = QuestionGenerator(
        provider_type=provider,
        question_type=question_type,
        n_questions=num_questions,
        output_type=OutputType.JSON,
    )

    # Generate questions
    result = question_generator.invoke(combined_text)

    # Format the result for display
    if question_type == QuestionType.QA:
        questions = result.get("open_ended_questions", [])
        formatted_output = "<h3>Generated Open-Ended Questions</h3>"
        for i, q in enumerate(questions, 1):
            formatted_output += (
                f"<div class='question'><b>Q{i}:</b> {q['question_prompt']}</div>"
            )
            formatted_output += (
                f"<div class='answer'><b>Answer:</b> {q['reference_answer']}</div><br>"
            )
    else:
        questions = result.get("mcq_questions", [])
        formatted_output = "<h3>Generated Multiple-Choice Questions</h3>"
        for i, q in enumerate(questions, 1):
            formatted_output += (
                f"<div class='question'><b>Q{i}:</b> {q['question_text']}</div>"
            )
            formatted_output += "<div class='options'><b>Options:</b><ul>"
            for opt in q["answer_options"]:
                correct = "✓ " if opt["option_id"] in q["correct_option_ids"] else ""
                formatted_output += (
                    f"<li>{correct}{opt['option_id']}: {opt['option_text']}</li>"
                )
            formatted_output += "</ul></div>"
            formatted_output += f"<div class='explanation'><b>Explanation:</b> {q['answer_explanation']}</div><br>"

    return formatted_output


def create_app():
    """Create and configure the Gradio application."""
    with gr.Blocks(
        title="Document QA Generator",
        css="""
        .question { margin-top: 10px; }
        .answer, .options, .explanation { margin-left: 20px; margin-top: 5px; }
        .options ul { margin-top: 0; }
        .context-path { font-size: 0.8em; color: #666; margin-bottom: 2px; }
        .chunk { border-bottom: 1px solid #eee; padding: 10px 0; }
    """,
    ) as app:
        gr.Markdown("# Document Question Generator")
        gr.Markdown("Upload a document to generate questions from its content.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Document", file_types=[".txt", ".md", ".pdf"]
                )
                header_filter = gr.Dropdown(
                    choices=["all", "1", "2", "3", "4", "none"],
                    value="all",
                    label="Filter by Header Level",
                )

                with gr.Row():
                    process_btn = gr.Button("Process Document")

                with gr.Accordion("Generation Settings", open=False):
                    provider_dropdown = gr.Dropdown(
                        choices=[
                            LLMProviderType.OLLAMA,
                            LLMProviderType.OPENAI,
                            LLMProviderType.ANTHROPIC,
                            LLMProviderType.XAI,
                        ],
                        value=LLMProviderType.OLLAMA,
                        label="LLM Provider",
                    )

                    question_type_dropdown = gr.Dropdown(
                        choices=[QuestionType.QA, QuestionType.MCQ],
                        value=QuestionType.QA,
                        label="Question Type",
                    )

                    num_questions = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of Questions",
                    )

                generate_btn = gr.Button("Generate Questions", variant="primary")
                questions_output = gr.HTML(label="Generated Questions")

            with gr.Column(scale=2):
                chunk_selector = gr.CheckboxGroup(
                    label="Select Chunks for Question Generation", choices=[]
                )
                chunks_output = gr.HTML(label="Document Chunks")

        # State variables to store processed data
        chunks_state = gr.State([])

        # Event handlers
        def update_ui(file, header_level):
            if file is None:
                return (
                    gr.HTML(value="Please upload a document first."),
                    gr.CheckboxGroup(choices=[]),
                    [],
                )

            chunk_data, _ = process_document(file)
            filtered_chunks = filter_chunks_by_header(chunk_data, header_level)

            # Create HTML display of chunks
            html_output = "<div class='chunks'>"
            for chunk in filtered_chunks:
                html_output += f"<div class='chunk' id='chunk-{chunk['id']}'>"
                html_output += format_chunk_for_display(chunk)
                html_output += "</div>"
            html_output += "</div>"

            # Update the chunk selector
            chunk_choices = [
                (f"Chunk {chunk['id']}", chunk["id"]) for chunk in filtered_chunks
            ]

            return (
                gr.HTML(value=html_output),
                gr.CheckboxGroup(choices=chunk_choices),
                chunk_data,
            )

        process_btn.click(
            fn=update_ui,
            inputs=[file_input, header_filter],
            outputs=[chunks_output, chunk_selector, chunks_state],
        )

        file_input.upload(
            fn=update_ui,
            inputs=[file_input, header_filter],
            outputs=[chunks_output, chunk_selector, chunks_state],
        )

        header_filter.change(
            fn=update_ui,
            inputs=[file_input, header_filter],
            outputs=[chunks_output, chunk_selector, chunks_state],
        )

        generate_btn.click(
            fn=generate_questions,
            inputs=[
                chunks_state,
                chunk_selector,
                question_type_dropdown,
                provider_dropdown,
                num_questions,
            ],
            outputs=[questions_output],
        )

    return app


def main():
    app = create_app()
    app.launch(share=False)


if __name__ == "__main__":
    main()
