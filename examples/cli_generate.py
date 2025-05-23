import argparse
import json
import logging
import os
import random
from typing import Dict, List, Tuple

from llama_index.core.schema import NodeRelationship

from qageneratorllm.generator import LLMProviderType, QuestionGenerator, QuestionType
from qageneratorllm.loader import chunk_document, sort_chunked_documents
from qageneratorllm.qa_dataclass import OutputType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def build_h1_hierarchy(all_chunks: List[Dict]) -> Dict[str, Dict]:
    """
    Build a hierarchical structure of H1 sections with all their descendants.
    Returns a dict where keys are H1 node_ids and values contain the H1 chunk and all descendants.
    """
    # Build node_id → chunk mapping
    node_id_to_chunk = {chunk["node_id"]: chunk for chunk in all_chunks}

    # Build children mapping
    node_id_to_children_map = {}
    for chunk in all_chunks:
        node_id_to_children_map[chunk["node_id"]] = chunk["children_ids"]

    # Find all H1 chunks
    h1_chunks = [c for c in all_chunks if c.get("header_level") == 1]

    h1_hierarchy = {}

    for h1_chunk in h1_chunks:
        h1_node_id = h1_chunk["node_id"]

        # Get all descendants of this H1
        all_descendants = _get_all_descendant_node_ids(
            h1_node_id, node_id_to_children_map
        )

        # Collect all chunks for this H1 section (H1 + descendants)
        section_chunks = [h1_chunk]
        for descendant_id in all_descendants:
            if descendant_id in node_id_to_chunk:
                section_chunks.append(node_id_to_chunk[descendant_id])

        # Sort chunks by their UI ID to maintain document order
        section_chunks.sort(key=lambda x: x["id"])

        h1_hierarchy[h1_node_id] = {
            "h1_chunk": h1_chunk,
            "all_chunks": section_chunks,
            "descendant_count": len(all_descendants),
            "total_text_length": sum(len(chunk["text"]) for chunk in section_chunks),
        }

    return h1_hierarchy


providers = {
    "OLLAMA": LLMProviderType.OLLAMA,
    "OPENAI": LLMProviderType.OPENAI,
    "ANTHROPIC": LLMProviderType.ANTHROPIC,
    "GROQ": LLMProviderType.GROQ,
    "XAI": LLMProviderType.XAI,
}


def get_args():
    parser = argparse.ArgumentParser(description="Generate questions from documents.")
    parser.add_argument(
        "--source", help="Path to folder or single file.", required=True
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=LLMProviderType.OLLAMA,
        help="Name of LLM provider. Options: " + ", ".join(providers.keys()),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use. Defaults to provider's default.",
    )
    parser.add_argument(
        "--qtype", type=str, default="QA", help="Question type, QA or MCQ."
    )
    parser.add_argument(
        "--num", type=int, default=3, help="Number of questions to generate."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Folder to save output."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of text chunks for processing.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks.",
    )
    # ...existing code for more CLI arguments...

    args = parser.parse_args()
    return args


def cli_main(args):
    # 1. Load and chunk documents
    source_path = args.source
    all_chunks = []

    # Check if the source is a directory or a file
    if os.path.isdir(source_path):
        # If it's a directory, iterate over all files in the directory
        for file_name in os.listdir(source_path):
            file_path = os.path.join(source_path, file_name)
            if os.path.isfile(file_path):
                chunk_data, _ = process_document(
                    file_path,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
                all_chunks.extend(chunk_data)
    elif os.path.isfile(source_path):
        # If it's a single file, process it directly
        chunk_data, _ = process_document(
            source_path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        all_chunks.extend(chunk_data)
    else:
        logger.error(f"Source not found: {source_path}")
        return

    logger.info(f"Total chunks processed: {len(all_chunks)}")

    # Build H1 hierarchy
    h1_hierarchy = build_h1_hierarchy(all_chunks)
    logger.info(f"Found {len(h1_hierarchy)} H1 sections:")

    for h1_node_id, h1_data in h1_hierarchy.items():
        h1_chunk = h1_data["h1_chunk"]
        descendant_count = h1_data["descendant_count"]
        total_chunks = len(h1_data["all_chunks"])
        logger.info(
            f"  - {h1_chunk.get('title', 'Untitled')} ({total_chunks} chunks, {descendant_count} descendants)"
        )

    if not h1_hierarchy:
        logger.error("No H1 headers found. Cannot generate questions.")
        return

    # Randomly select H1 sections
    h1_keys = list(h1_hierarchy.keys())
    num_to_select = min(len(h1_keys), args.num)
    selected_h1_keys = random.sample(h1_keys, num_to_select)

    logger.info(f"\nRandomly selected {num_to_select} H1 sections:")

    # Collect all chunks from selected H1 sections
    selected_chunks = []
    for h1_key in selected_h1_keys:
        h1_data = h1_hierarchy[h1_key]
        h1_chunk = h1_data["h1_chunk"]
        section_chunks = h1_data["all_chunks"]
        selected_chunks.extend(section_chunks)
        logger.info(
            f"  - {h1_chunk.get('title', 'Untitled')} ({len(section_chunks)} chunks)"
        )

    # Sort all selected chunks by ID to maintain document order
    selected_chunks.sort(key=lambda x: x["id"])
    combined_text = "\n\n".join(c["text"] for c in selected_chunks)

    logger.info(f"\nTotal selected chunks: {len(selected_chunks)}")
    logger.info(f"Combined text length: {len(combined_text)} characters")
    logger.info("Generating questions...")

    # 5. Generate questions
    provider_type = getattr(
        LLMProviderType, args.provider.upper(), LLMProviderType.OLLAMA
    )
    question_type = getattr(QuestionType, args.qtype.upper(), QuestionType.QA)
    output_type = OutputType.DATACLASS
    question_generator = QuestionGenerator(
        provider_type=provider_type,
        question_type=question_type,
        n_questions=args.num,
        model_name=args.model_name,
        output_type=output_type,
    )
    result = question_generator.invoke(combined_text)

    # 6. Save output to the specified folder
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "generated_questions.json")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                result.model_dump() if output_type == OutputType.DATACLASS else result,
                indent=2,
            )
        )

    logger.info(f"Generated questions saved to: {output_file}")


if __name__ == "__main__":
    args = get_args()
    cli_main(args)
