from typing import Dict, List, Tuple
import argparse
import os
import random
import json

from qageneratorllm.generator import LLMProviderType, QuestionGenerator, QuestionType
from qageneratorllm.loader import chunk_document, sort_chunked_documents
from qageneratorllm.qa_dataclass import OutputType


def process_document(
    file_path, chunk_size=1000, chunk_overlap=200
) -> Tuple[List[Dict], List[str]]:
    """Process uploaded document and return chunks with metadata."""
    if file_path is None:
        return [], []

    # Chunk the document and organize by relationships
    chunked_doc = chunk_document(
        file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    organized_chunks = sort_chunked_documents(chunked_doc)

    all_nodes: List[str] = []
    chunk_data: List[Dict] = []
    ui_id_counter = 0

    for _, nodes_in_group in organized_chunks.items():
        for node in nodes_in_group:
            all_nodes.append(node)

            header_level = node.metadata.get("level")
            header_tag = f"h{header_level}" if header_level else "p"

            parent_relation = node.relationships.get(
                "4", {}
            )  # "4" typically indicates parent
            parent_id = (
                parent_relation.get("node_id")
                if isinstance(parent_relation, dict)
                else None
            )

            children_relation = node.relationships.get(
                "5", []
            )  # "5" typically indicates children
            children_ids = (
                [
                    child.get("node_id")
                    for child in children_relation
                    if isinstance(child, dict) and child.get("node_id")
                ]
                if isinstance(children_relation, list)
                else []
            )

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
    visited_in_bfs = {start_node_id}

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


def cli_main():
    parser = argparse.ArgumentParser(description="Generate questions from documents.")
    parser.add_argument("--source", help="Path to folder or single file.", required=True)
    parser.add_argument("--provider", type=str, default="OLLAMA", help="Name of LLM provider.")
    parser.add_argument("--qtype", type=str, default="QA", help="Question type, QA or MCQ.")
    parser.add_argument("--num", type=int, default=3, help="Number of questions to generate.")
    parser.add_argument("--output_dir", type=str, default=".", help="Folder to save output.")
    # ...existing code for more CLI arguments...

    args = parser.parse_args()

    # 1. Load and chunk documents
    source_path = args.source
    all_chunks = []

    # Check if the source is a directory or a file
    if os.path.isdir(source_path):
        # If it's a directory, iterate over all files in the directory
        for file_name in os.listdir(source_path):
            file_path = os.path.join(source_path, file_name)
            if os.path.isfile(file_path):
                chunk_data, _ = process_document(file_path)
                all_chunks.extend(chunk_data)
    elif os.path.isfile(source_path):
        # If it's a single file, process it directly
        chunk_data, _ = process_document(source_path)
        all_chunks.extend(chunk_data)
    else:
        print(f"Source not found: {source_path}")
        return

    # Build node_id → children map
    node_id_to_children_map: Dict[str, List[str]] = {}
    for chunk in all_chunks:
        node_id_to_children_map[chunk["node_id"]] = chunk["children_ids"]

    # Filter chunks by header_level=1
    h1_chunks = [c for c in all_chunks if c.get("header_level") == 1]

    # Randomly select the specified number from these h1 chunks
    selected_chunks = random.sample(h1_chunks, min(len(h1_chunks), args.num))

    # Gather all selected node_ids and their descendants
    selected_node_ids = set()
    for chunk in selected_chunks:
        selected_node_ids.add(chunk["node_id"])
        descendants = _get_all_descendant_node_ids(chunk["node_id"], node_id_to_children_map)
        selected_node_ids.update(descendants)

    # Collect relevant chunks' text
    relevant_chunks = [c for c in all_chunks if c["node_id"] in selected_node_ids]
    combined_text = "\n\n".join(c["text"] for c in relevant_chunks)

    # 5. Generate questions
    provider_type = getattr(LLMProviderType, args.provider.upper(), LLMProviderType.OLLAMA)
    question_type = getattr(QuestionType, args.qtype.upper(), QuestionType.QA)
    question_generator = QuestionGenerator(
        provider_type=provider_type,
        question_type=question_type,
        n_questions=args.num,
        output_type=OutputType.DATACLASS,
    )
    result = question_generator.invoke(combined_text)

    # 6. Save output to the specified folder
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "generated_questions.json")

    with open(output_file, "w") as f:
        f.write(json.dumps(result.model_dump(), indent=2))

    print(f"Generated questions saved to: {output_file}")


def main():
    cli_main()


if __name__ == "__main__":
    main()