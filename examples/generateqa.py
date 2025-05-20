from typing import Dict, List, Tuple

import gradio as gr
from llama_index.core.schema import TextNode

from qageneratorllm.generator import LLMProviderType, QuestionGenerator, QuestionType
from qageneratorllm.loader import chunk_document, sort_chunked_documents
from qageneratorllm.qa_dataclass import OutputType


def process_document(
    file_path, chunk_size=1000, chunk_overlap=200
) -> Tuple[List[Dict], List[TextNode]]:
    """Process uploaded document and return chunks with metadata."""
    if file_path is None:
        return [], []

    # Chunk the document and organize by relationships
    chunked_doc = chunk_document(
        file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
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


def _get_all_descendant_node_ids(
    start_node_id: str, node_id_to_children_map: Dict[str, List[str]]
) -> set[str]:
    """
    Recursively get all descendant node IDs for a given starting node_id.
    """
    descendants = set()
    queue = list(node_id_to_children_map.get(start_node_id, []))
    visited_in_bfs = {
        start_node_id
    }  # Keep track of nodes visited in this BFS traversal

    while queue:
        current_child_id = queue.pop(0)
        if current_child_id in visited_in_bfs:
            continue
        visited_in_bfs.add(current_child_id)
        descendants.add(current_child_id)

        # Add grandchildren to the queue
        grandchildren = node_id_to_children_map.get(current_child_id, [])
        for grandchild_id in grandchildren:
            if grandchild_id not in visited_in_bfs:
                queue.append(grandchild_id)
    return descendants


def generate_questions(
    chunks: List[Dict],  # This is the full chunks_state (list of chunk_data dicts)
    selected_chunk_ui_ids: List[int],  # List of UI integer IDs selected by the user
    question_type: str,
    provider: str,
    num_questions: int,
) -> str:
    """Generate questions for selected chunks and all their descendants."""
    if not selected_chunk_ui_ids:
        return "Please select at least one chunk to generate questions."

    # Build a map from original node_id to its children_ids for efficient lookup
    node_id_to_children_map: Dict[str, List[str]] = {}
    for chunk_dict in chunks:
        current_node_id = chunk_dict["node_id"]
        if current_node_id not in node_id_to_children_map:
            node_id_to_children_map[current_node_id] = chunk_dict["children_ids"]
        # Assuming children_ids are consistent for all splits of the same original node_id

    all_node_ids_for_generation = set()

    # For each UI ID selected by the user:
    for ui_id in selected_chunk_ui_ids:
        # Find the corresponding chunk dictionary
        selected_chunk_dict = next((cd for cd in chunks if cd["id"] == ui_id), None)
        if not selected_chunk_dict:
            continue

        original_node_id = selected_chunk_dict["node_id"]
        all_node_ids_for_generation.add(original_node_id)

        # Get all descendants of this original_node_id
        descendants = _get_all_descendant_node_ids(
            original_node_id, node_id_to_children_map
        )
        all_node_ids_for_generation.update(descendants)

    # Collect texts from all relevant nodes (selected + descendants), including all their splits
    relevant_texts_with_ui_id = []
    for chunk_dict in chunks:
        if chunk_dict["node_id"] in all_node_ids_for_generation:
            relevant_texts_with_ui_id.append((chunk_dict["id"], chunk_dict["text"]))

    # Sort by the original UI ID to maintain document order
    relevant_texts_with_ui_id.sort(key=lambda x: x[0])

    selected_texts = [text for _, text in relevant_texts_with_ui_id]

    if not selected_texts:
        return "No text found for the selected chunks and their descendants."

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
                            LLMProviderType.GROQ,
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

                with gr.Accordion("Chunking Settings", open=False):
                    chunk_size = gr.Slider(
                        minimum=200,
                        maximum=2000,
                        value=1000,
                        step=100,
                        label="Chunk Size",
                    )

                    chunk_overlap = gr.Slider(
                        minimum=0,
                        maximum=500,
                        value=200,
                        step=50,
                        label="Chunk Overlap",
                    )

                generate_btn = gr.Button("Generate Questions", variant="primary")
                questions_output = gr.HTML(label="Generated Questions")

            with gr.Column(scale=2):
                chunk_selector = gr.Dropdown(
                    label="Select Chunks for Question Generation",
                    choices=[],
                    multiselect=True,
                    info="Select one or more chunks to generate questions from",
                )
                chunks_output = gr.HTML(label="Document Chunks")

        # State variables to store processed data
        chunks_state = gr.State([])

        # Event handlers
        def update_ui(file, header_level, size, overlap):
            if file is None:
                return (
                    gr.HTML(value="Please upload a document first."),
                    gr.Dropdown(choices=[]),
                    [],
                )

            chunk_data, _ = process_document(
                file, chunk_size=size, chunk_overlap=overlap
            )
            filtered_chunks = filter_chunks_by_header(chunk_data, header_level)

            # Create HTML display of chunks
            html_output = "<div class='chunks'>"
            for chunk in filtered_chunks:
                html_output += f"<div class='chunk' id='chunk-{chunk['id']}'>"
                html_output += format_chunk_for_display(chunk)
                html_output += "</div>"
            html_output += "</div>"

            # Update the chunk selector with more descriptive labels
            chunk_choices = []
            for chunk in filtered_chunks:
                label = f"Chunk {chunk['id']}"
                if chunk.get("header_level"):
                    label += f" (H{chunk['header_level']})"
                if chunk.get("context_path"):
                    path_preview = (
                        chunk["context_path"].split(" > ")[-1]
                        if " > " in chunk["context_path"]
                        else chunk["context_path"]
                    )
                    label += f": {path_preview}"
                chunk_choices.append((label, chunk["id"]))

            return (
                gr.HTML(value=html_output),
                gr.Dropdown(choices=chunk_choices),
                chunk_data,
            )

        process_btn.click(
            fn=update_ui,
            inputs=[file_input, header_filter, chunk_size, chunk_overlap],
            outputs=[chunks_output, chunk_selector, chunks_state],
        )

        file_input.upload(
            fn=update_ui,
            inputs=[file_input, header_filter, chunk_size, chunk_overlap],
            outputs=[chunks_output, chunk_selector, chunks_state],
        )

        header_filter.change(
            fn=update_ui,
            inputs=[file_input, header_filter, chunk_size, chunk_overlap],
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
