from collections import defaultdict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.schema import TextNode
from node_chunker.chunks import chunk_document_by_toc_to_text_nodes


def _split_texts(
    node: TextNode, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[TextNode]:
    if node.text is None or len(node.text) < chunk_size + chunk_overlap:
        node.metadata["chunk_id"] = node.node_id + "_0"
        return [node]

    def make_copy(text: str, pos: int) -> TextNode:
        """
        Create a copy of the TextNode to avoid modifying the original.
        """
        cp_node = node.model_copy(deep=True)
        cp_node.metadata["chunk_id"] = f"{pos}"
        cp_node.text = text
        return cp_node

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = splitter.split_text(node.text)
    return [make_copy(chunk, pos) for pos, chunk in enumerate(splits)]


def chunk_document(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Chunk a document into smaller pieces based on the table of contents (TOC).

    Args:
        file_path (str): Path to the document file to be chunked.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list[TextNode]: List of TextNode containing the chunked text and its metadata.
    """
    return [
        _split_texts(chunk, chunk_size, chunk_overlap)
        for chunk in chunk_document_by_toc_to_text_nodes(str(file_path))
    ]


def sort_chunked_documents(
    documents: list[list[TextNode]],
) -> dict[str, list[TextNode]]:
    results: dict[str, list[TextNode]] = defaultdict(list)
    for doc in documents:
        for chunk in doc:
            results[chunk.node_id].append(chunk)
    results = {
        k: sorted(v, key=lambda x: x.metadata["chunk_id"]) for k, v in results.items()
    }
    return results


def chunk_documents(
    file_paths: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Chunk multiple documents into smaller pieces.

    Args:
        file_paths (list[str]): List of paths to document files.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list[list[list[TextNode]]]: List of chunked documents.
    """
    return [
        chunk_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for path in file_paths
    ]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Chunk documents into smaller pieces")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing text files to be chunked",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder where chunked files will be saved",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of each chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks (default: 200)",
    )

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    output_folder.mkdir(parents=True, exist_ok=True)

    for file_path in input_folder.rglob("*"):
        chunked_documents = sort_chunked_documents(
            chunk_document(
                file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        )
        output_file_path = output_folder / (file_path.stem + ".json")
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            json.dump(
                {
                    k: [doc.model_dump() for doc in chunks]
                    for k, chunks in chunked_documents.items()
                },
                out_f,
                ensure_ascii=False,
                indent=2,
            )
