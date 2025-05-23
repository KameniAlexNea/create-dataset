from typing import Dict, List, Tuple

import gradio as gr
from llama_index.core.schema import NodeRelationship, TextNode

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

    all_nodes: List[TextNode] = []
    chunk_data: List[Dict] = []
    ui_id_counter = 0

    for _, nodes_in_group in organized_chunks.items():
        for node in nodes_in_group:
            all_nodes.append(node)

            header_level = node.metadata.get("level")
            header_tag = f"h{header_level}" if header_level else "p"

            parent_relation = node.relationships.get(NodeRelationship.PARENT, {})
            parent_id = (
                parent_relation.node_id if hasattr(parent_relation, "node_id") else None
            )

            children_relation = node.relationships.get(NodeRelationship.CHILD, [])
            children_ids = []
            if isinstance(children_relation, list):
                children_ids = [
                    child.node_id
                    for child in children_relation
                    if hasattr(child, "node_id")
                ]
            elif hasattr(children_relation, "node_id"):
                children_ids = [children_relation.node_id]

            context_path = node.metadata.get("context", "")

            chunk_data.append(
                {
                    "id": ui_id_counter,
                    "node_id": node.node_id,
                    "text": node.text,
                    "header_level": header_level,
                    "header_tag": header_tag,
                    "parent_id": parent_id,
                    "children_ids": children_ids,
                    "context_path": context_path,
                    "metadata": node.metadata,
                    "title": node.metadata.get("title", f"Chunk {ui_id_counter}"),
                }
            )
            ui_id_counter += 1

    return chunk_data, all_nodes


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
    """
    Return a Markdown string for the chunk, removing HTML.
    """
    header_level = chunk.get("header_level") or 1
    heading_symbol = "#" * max(1, min(header_level, 6))
    context_str = (
        f"**Context:** {chunk['context_path']}\n\n" if chunk.get("context_path") else ""
    )
    chunk_id = chunk["id"]
    return (
        f"{heading_symbol} {chunk.get('title', f'Chunk {chunk_id}')}\n\n"
        f"{context_str}{chunk.get('text', '')}\n"
    )


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
    chunks: List[Dict],
    selected_chunk_ui_ids: List[int],
    question_type: str,
    provider: str,
    num_questions: int,
) -> Tuple[str, str]:
    """Generate questions for selected chunks and all their descendants."""
    if not selected_chunk_ui_ids:
        return (
            "Please select at least one chunk to generate questions.",
            "No chunks selected.",
        )

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
        return (
            "No text found for the selected chunks and their descendants.",
            "No relevant text found.",
        )

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

    # Format the result as Markdown
    questions_markdown = []
    if question_type == QuestionType.QA:
        questions = result.get("open_ended_questions", [])
        questions_markdown.append("### Generated Open-Ended Questions")
        questions_markdown.append("---")
        for i, q in enumerate(questions, 1):
            questions_markdown.append(f"**Q{i}:** {q['question_prompt']}")
            questions_markdown.append(f"**Answer:** {q['reference_answer']}")
            questions_markdown.append("---")
    else:
        questions = result.get("mcq_questions", [])
        questions_markdown.append("### Generated Multiple-Choice Questions")
        questions_markdown.append("---")
        for i, q in enumerate(questions, 1):
            questions_markdown.append(f"**Q{i}:** {q['question_text']}")
            questions_markdown.append("**Options:**")
            for opt in q["answer_options"]:
                correct_indicator = (
                    " (Correct)" if opt["option_id"] in q["correct_option_ids"] else ""
                )
                questions_markdown.append(
                    f"- {opt['option_id']}: {opt['option_text']}{correct_indicator}"
                )
            questions_markdown.append(f"**Explanation:** {q['answer_explanation']}")
            questions_markdown.append("---")

    formatted_questions_output = "\n\n".join(questions_markdown)

    # Format the used chunks as Markdown instead of HTML
    used_chunks_md_parts = ["### Context Used for Generation:\n"]
    chunks_by_ui_id = {chunk["id"]: chunk for chunk in chunks}

    # Get unique UI IDs from relevant_texts_with_ui_id, maintaining order
    ordered_used_ui_ids = []
    seen_ui_ids = set()
    for ui_id, _ in relevant_texts_with_ui_id:
        if ui_id not in seen_ui_ids:
            ordered_used_ui_ids.append(ui_id)
            seen_ui_ids.add(ui_id)

    for ui_id in ordered_used_ui_ids:
        chunk_to_display = chunks_by_ui_id.get(ui_id)
        if chunk_to_display:
            used_chunks_md_parts.append(format_chunk_for_display(chunk_to_display))
            used_chunks_md_parts.append("---\n")
    formatted_used_chunks_markdown = "\n".join(used_chunks_md_parts)

    return formatted_questions_output, formatted_used_chunks_markdown


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
        .chunk-separator { margin: 5px 0; border: none; border-top: 1px dashed #ccc; }
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
                questions_output = gr.Markdown(label="Generated Questions")
                used_chunks_display = gr.Markdown(label="Context Used for Generation")

            with gr.Column(scale=2):
                chunk_selector = gr.Dropdown(
                    label="Select Chunks for Question Generation",
                    choices=[],
                    multiselect=True,
                    info="Select one or more chunks to generate questions from",
                )
                chunks_output = gr.Markdown(label="Document Chunks")

        # State variables to store processed data
        chunks_state = gr.State([])

        # Event handlers
        def update_ui(file, header_level, size, overlap):
            if file is None:
                return (
                    gr.Markdown(value="Please upload a document first."),
                    gr.Dropdown(choices=[]),
                    [],
                )

            chunk_data, _ = process_document(
                file, chunk_size=size, chunk_overlap=overlap
            )
            filtered_chunks = filter_chunks_by_header(chunk_data, header_level)

            # Build Markdown output for filtered chunks
            chunks_markdown = ["# Document Chunks\n"]
            for chunk in filtered_chunks:
                chunks_markdown.append(format_chunk_for_display(chunk))
                chunks_markdown.append("---\n")
            markdown_output = "\n".join(chunks_markdown)

            # Prepare chunk selector labels
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
                gr.Markdown(value=markdown_output),
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
            outputs=[questions_output, used_chunks_display],
        )

    return app


def main():
    app = create_app()
    app.launch(share=False)


if __name__ == "__main__":
    main()
