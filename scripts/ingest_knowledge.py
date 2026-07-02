#!/usr/bin/env python3
"""
scripts/ingest_knowledge.py
─────────────────────────────────────────────────────────────────────────────
Standalone CLI tool for pre-populating the FAISS knowledge base index
without starting the Streamlit application.

Use this script to:
  • Batch-ingest a directory of documents before first launch
  • Add a single new document to an existing index
  • Rebuild the index from scratch

Usage
─────
    # Ingest the default knowledge base directory
    python scripts/ingest_knowledge.py

    # Ingest a custom directory
    python scripts/ingest_knowledge.py --dir /path/to/docs

    # Add a single file to the existing index
    python scripts/ingest_knowledge.py --file /path/to/report.pdf

    # Wipe the existing index and rebuild from default KB directory
    python scripts/ingest_knowledge.py --rebuild

    # Wipe and rebuild from a custom directory
    python scripts/ingest_knowledge.py --rebuild --dir /path/to/docs

Requirements
────────────
    ANTHROPIC_API_KEY must be set in .env (needed for RAG engine init).
    The embedding model runs locally — no extra API calls for ingestion.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# Add src/ to sys.path so rag_engine imports work
sys.path.insert(0, str(ROOT / "src"))

# Set CWD to project root so relative paths inside ChurnRAGEngine resolve
import os
os.chdir(ROOT)

# ── Import ────────────────────────────────────────────────────────────────
from rag_engine import ChurnRAGEngine


# ── Helpers ───────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 58
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _print_stats(engine: ChurnRAGEngine) -> None:
    stats = engine.get_index_stats()
    status = "READY" if stats["ready"] else "EMPTY"
    print(f"\n  Index status  : {status}")
    print(f"  Total vectors : {stats['total_vectors']}")
    print(f"  Index path    : {engine._faiss_dir}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChurnGuard AI — FAISS knowledge base ingestion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--dir",
        metavar="DIRECTORY",
        default=str(ROOT / "data" / "knowledge_base"),
        help=(
            "Directory of documents to ingest. "
            "Recursively finds CSV, PDF, TXT, MD files. "
            "(default: data/knowledge_base/)"
        ),
    )
    source_group.add_argument(
        "--file",
        metavar="FILE",
        help="Path to a single file to add to the existing index.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing FAISS index before ingesting (full rebuild).",
    )

    args = parser.parse_args()

    _banner("ChurnGuard AI — Knowledge Base Ingestion")

    # ── Optional rebuild ───────────────────────────────────────────────────
    if args.rebuild:
        idx_path = ROOT / "models" / "faiss_index"
        if idx_path.exists():
            shutil.rmtree(idx_path)
            print(f"\n  [rebuild] Cleared existing index at {idx_path}")
        else:
            print("\n  [rebuild] No existing index found — starting fresh")

    # ── Initialise engine ──────────────────────────────────────────────────
    print("\n  Initialising RAG engine …")
    print("  (First run downloads the embedding model ~90 MB — please wait)")
    t0 = time.time()

    try:
        engine = ChurnRAGEngine()
    except ValueError as exc:
        print(f"\n  ❌  Configuration error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n  ❌  Failed to initialise engine: {exc}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"  Engine ready in {elapsed:.1f}s")

    # ── Ingest ────────────────────────────────────────────────────────────
    n_chunks = 0

    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            print(f"\n  ❌  File not found: {fp}")
            sys.exit(1)
        print(f"\n  Adding file: {fp.name}")
        t1 = time.time()
        try:
            n_chunks = engine.add_documents_from_upload(fp.read_bytes(), fp.name)
        except Exception as exc:
            print(f"\n  ❌  Ingestion failed: {exc}")
            sys.exit(1)
        print(f"  Done in {time.time() - t1:.1f}s")

    else:
        d = Path(args.dir)
        if not d.exists():
            print(f"\n  ❌  Directory not found: {d}")
            sys.exit(1)

        # Count files before ingesting
        supported = {".csv", ".pdf", ".txt", ".md"}
        files_found = [f for f in d.rglob("*")
                       if f.is_file() and f.suffix.lower() in supported]
        if not files_found:
            print(f"\n  ❌  No supported documents found in {d}")
            print(f"       Supported types: {', '.join(sorted(supported))}")
            sys.exit(1)

        print(f"\n  Ingesting directory: {d}")
        print(f"  Files found: {len(files_found)}")
        for f in files_found:
            print(f"    • {f.name}")

        t1 = time.time()
        try:
            n_chunks = engine.build_index_from_directory(str(d))
        except Exception as exc:
            print(f"\n  ❌  Ingestion failed: {exc}")
            sys.exit(1)
        print(f"  Done in {time.time() - t1:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  ✅  {n_chunks} chunk(s) indexed")
    _print_stats(engine)

    print("\n  Run the Streamlit app to start querying:")
    print("    streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
