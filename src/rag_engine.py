import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator, Optional

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()


# ── Ollama health check ───────────────────────────────────────────────────

def _verify_ollama(base_url: str, llm_model: str, embed_model: str) -> None:
    """
    Confirm the Ollama server is reachable and both required models are pulled.
    Uses only stdlib urllib — no extra dependencies.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {base_url}.\n"
            "  Fix → Install Ollama : https://ollama.ai\n"
            "  Fix → Start server   : ollama serve"
        ) from exc

    available  = [m["name"] for m in payload.get("models", [])]
    missing    = []

    for model in (llm_model, embed_model):
        base = model.split(":")[0]
        if not any(m.startswith(base) for m in available):
            missing.append(model)

    if missing:
        cmds = "\n".join(f"  Fix → ollama pull {m}" for m in missing)
        raise RuntimeError(
            f"Missing Ollama model(s): {', '.join(missing)}\n"
            f"{cmds}\n"
            f"  Available: {', '.join(available) or 'none'}"
        )


# ── RAG prompt ────────────────────────────────────────────────────────────

_RAG_SYSTEM = """\
You are a senior customer intelligence analyst at a telecom company.
You have access to the company's customer support records, CRM interaction
notes, and complaint history — loaded below as retrieved documents.

IMPORTANT RULES:
1. Answer ONLY using information found in the retrieved documents.
2. If the answer is not present, say clearly:
   "I couldn't find that information in the current knowledge base."
3. Never fabricate customer IDs, dates, agent names, or ticket details.
4. Be specific — reference actual IDs and agent names when present in the docs.
5. Use bullet points for lists. Keep answers clear and actionable.
6. If multiple customers match, summarise patterns rather than individual records.

Retrieved Documents:
────────────────────
{context}
────────────────────\
"""

_RAG_HUMAN = "{question}"

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _RAG_SYSTEM),
    ("human",  _RAG_HUMAN),
])


# ── Engine ────────────────────────────────────────────────────────────────

class ChurnRAGEngine:
    """
    Full RAG pipeline: ingest → embed (local) → store → retrieve → generate (local).

    Both the embedding model and the LLM run on the local Ollama server.
    The FAISS index is saved to disk — no re-embedding on restart.

    Raises RuntimeError at construction if Ollama is not running or
    required models have not been pulled, so the Streamlit app can
    degrade gracefully with a helpful error message.
    """

    _SUPPORTED_EXT   = {".csv", ".pdf", ".md"}
    _DEFAULT_BASE_URL  = "http://localhost:11434"
    _DEFAULT_LLM       = "llama3.1:8b"
    _DEFAULT_EMBED     = "nomic-embed-text"
    _DEFAULT_MAX_TOK   = 1024
    _CHUNK_SIZE        = 2000
    _CHUNK_OVERLAP     = 100
    _TOP_K             = 8
    _DEFAULT_EMBED_BATCH_SIZE = 100   # chunks per Ollama /api/embed call
    _DEFAULT_EMBED_MAX_RETRIES = 3   # retries per batch before giving up

    def __init__(self) -> None:
        base_url    = os.getenv("OLLAMA_BASE_URL",  self._DEFAULT_BASE_URL)
        llm_model   = os.getenv("CHURN_LLM_MODEL",  self._DEFAULT_LLM)
        embed_model = os.getenv("CHURN_EMBED_MODEL", self._DEFAULT_EMBED)
        max_t       = int(os.getenv("CHURN_MAX_TOKENS", self._DEFAULT_MAX_TOK))

        self._faiss_dir = Path(os.getenv("CHURN_FAISS_DIR", "models/faiss_index"))
        self._kb_dir    = Path(os.getenv("CHURN_KB_DIR",    "data/knowledge_base"))

        # Embedding batch size — large single-shot embed_documents() calls
        # (e.g. all 18,994 chunks at once) can overwhelm Ollama's embedding
        # server on some platforms. Batching keeps each call small and adds
        # retry-with-backoff for transient failures. Tune via env if needed.
        self._embed_batch_size = int(
            os.getenv("CHURN_EMBED_BATCH_SIZE", self._DEFAULT_EMBED_BATCH_SIZE)
        )
        self._embed_max_retries = int(
            os.getenv("CHURN_EMBED_MAX_RETRIES", self._DEFAULT_EMBED_MAX_RETRIES)
        )

        # Fail fast with actionable error if Ollama isn't ready
        _verify_ollama(base_url, llm_model, embed_model)

        print(f"[rag] Embedding model : {embed_model}  (local via Ollama)")
        print(f"[rag] LLM model       : {llm_model}  (local via Ollama)")

        # Both models run locally via Ollama — zero API calls
        self._embeddings = OllamaEmbeddings(
            model=embed_model,
            base_url=base_url,
        )

        self._llm = ChatOllama(
            model=llm_model,
            base_url=base_url,
            temperature=0.1,      # Low temperature for factual retrieval answers
            num_predict=max_t,
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=self._CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

        self._vectorstore: Optional[FAISS] = None
        self._retriever   = None

        self._initialise_index()

    # ── Index lifecycle ───────────────────────────────────────────────────

    def _initialise_index(self) -> None:
        """Try to load persisted index; auto-ingest KB_DIR if not found."""
        if self._load_index():
            return
        if self._kb_dir.exists():
            files = [
                f for f in self._kb_dir.rglob("*")
                if f.is_file() and f.suffix in self._SUPPORTED_EXT
            ]
            if files:
                print(f"[rag] No saved index — building from {self._kb_dir}")
                self.build_index_from_directory(str(self._kb_dir))
            else:
                print(f"[rag] No documents in {self._kb_dir} — index empty")
        else:
            print("[rag] Knowledge base directory not found — index empty")

    def _load_index(self) -> bool:
        """Load FAISS index from disk. Returns True on success."""
        idx_file = self._faiss_dir / "index.faiss"
        if not idx_file.exists():
            return False
        try:
            self._vectorstore = FAISS.load_local(
                str(self._faiss_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._attach_retriever()
            n = int(self._vectorstore.index.ntotal)
            print(f"[rag] Loaded FAISS index: {n} vectors from {self._faiss_dir}")
            return True
        except Exception as exc:
            print(f"[rag] Index load failed ({exc}) — will rebuild")
            return False

    def _save_index(self) -> None:
        if self._vectorstore is None:
            return
        self._faiss_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self._faiss_dir))
        self._attach_retriever()

    def _attach_retriever(self) -> None:
        if self._vectorstore is not None:
            self._retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self._TOP_K},
            )

    # ── Document loading ──────────────────────────────────────────────────

    def _load_file(self, path: str) -> list[Document]:
        """Load one supported file. Returns [] on failure."""
        p   = Path(path)
        ext = p.suffix.lower()
        try:
            if ext == ".csv":
                loader = CSVLoader(str(p), encoding="utf-8",
                                   csv_args={"delimiter": ","})
            elif ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext in (".txt", ".md"):
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                return []
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = p.name   # normalise to filename only
            return docs
        except Exception as exc:
            print(f"[rag] ⚠️  Cannot load {p.name}: {exc}")
            return []

    def _load_directory(self, directory: str) -> list[Document]:
        docs: list[Document] = []
        for f in Path(directory).rglob("*"):
            if f.is_file() and f.suffix.lower() in self._SUPPORTED_EXT:
                docs.extend(self._load_file(str(f)))
        return docs

    # ── Batched embedding (production-safe ingestion) ─────────────────────
    #
    # Root cause this section fixes: the old code called
    #   FAISS.from_documents(all_chunks, self._embeddings)
    # which sends every chunk to OllamaEmbeddings.embed_documents() in ONE
    # HTTP request to Ollama's /api/embed endpoint. At scale (1,000+ chunks)
    # this single oversized request can crash Ollama's internal tokenizer
    # subprocess on some platforms — observed failure:
    #   Post "http://127.0.0.1:<random_port>/tokenize": connectex: No
    #   connection could be made because the target machine actively
    #   refused it.
    # Splitting into small batches keeps each HTTP call within a size
    # Ollama handles reliably, and retry-with-backoff absorbs transient
    # failures without aborting the whole ingestion run.

    def _embed_batch_with_retry(self, batch: list[Document]) -> None:
        """
        Embed one batch and add it to self._vectorstore, retrying with
        exponential backoff on transient failure. Creates the vectorstore
        on the first batch if it doesn't exist yet; otherwise appends.
        """
        delay = 1.0
        for attempt in range(1, self._embed_max_retries + 1):
            try:
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents(batch, self._embeddings)
                else:
                    self._vectorstore.add_documents(batch)
                return
            except Exception as exc:
                if attempt == self._embed_max_retries:
                    raise RuntimeError(
                        f"Embedding batch failed after {self._embed_max_retries} "
                        f"attempts (batch size={len(batch)}): {exc}"
                    ) from exc
                print(
                    f"[rag]   ⚠️  embed attempt {attempt}/{self._embed_max_retries} "
                    f"failed ({exc}) — retrying in {delay:.0f}s …"
                )
                time.sleep(delay)
                delay *= 2

    def _embed_documents_in_batches(self, chunks: list[Document], label: str) -> None:
        """
        Embed `chunks` into self._vectorstore in batches of
        self._embed_batch_size, logging progress as it goes.
        Does NOT reset self._vectorstore — caller decides overwrite vs. append.
        """
        total      = len(chunks)
        batch_size = self._embed_batch_size
        n_batches  = (total + batch_size - 1) // batch_size

        print(
            f"[rag] Embedding {total} chunks in {n_batches} batch(es) "
            f"of up to {batch_size} — {label}"
        )

        for i in range(0, total, batch_size):
            batch     = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(
                f"[rag]   batch {batch_num}/{n_batches} "
                f"({len(batch)} chunks, {i + len(batch)}/{total} total) …"
            )
            self._embed_batch_with_retry(batch)

        print(f"[rag] Embedding complete: {total} chunks — {label}")

    # ── Public ingestion API ──────────────────────────────────────────────

    def build_index_from_directory(self, directory: str) -> int:
        """
        (Re)build the full FAISS index from all supported files in *directory*.
        Overwrites any existing index on disk.
        Returns the number of chunks embedded and stored.
        """
        docs = self._load_directory(directory)
        if not docs:
            print(f"[rag] No supported documents in {directory}")
            return 0

        chunks = self._splitter.split_documents(docs)
        n_sources = len({d.metadata.get("source") for d in docs})

        # Reset before embedding so a rebuild truly overwrites — without
        # this, _embed_documents_in_batches() would append to whatever
        # vectorstore was already loaded in memory instead of replacing it.
        self._vectorstore = None

        self._embed_documents_in_batches(
            chunks, label=f"{n_sources} file(s) from {directory}"
        )
        self._save_index()

        print(f"[rag] Built index: {len(chunks)} chunks from {n_sources} file(s)")
        return len(chunks)

    def add_documents_from_upload(self, file_bytes: bytes, filename: str) -> int:
        """
        Incrementally embed and merge one uploaded file into the existing index.
        Returns the number of new chunks added.
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in self._SUPPORTED_EXT:
            raise ValueError(
                f"Unsupported type '{suffix}'. "
                f"Accepted: {', '.join(sorted(self._SUPPORTED_EXT))}"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            docs = self._load_file(tmp_path)
            if not docs:
                return 0

            for doc in docs:
                doc.metadata["source"] = filename   # restore original name

            chunks = self._splitter.split_documents(docs)

            # Appends to self._vectorstore if it exists, creates it on the
            # first batch otherwise — same batching/retry path as a full
            # rebuild, just without the reset-to-None step above.
            self._embed_documents_in_batches(chunks, label=f"'{filename}'")
            self._save_index()
            return len(chunks)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Context formatting ────────────────────────────────────────────────

    @staticmethod
    def _fmt_context(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            src  = doc.metadata.get("source", "unknown")
            text = doc.page_content.strip()
            parts.append(f"[Document {i} — source: {src}]\n{text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _enrich(question: str, ctx: Optional[dict]) -> str:
        if not ctx:
            return question
        header = (
            f"[Customer context: "
            f"Contract={ctx.get('Contract','?')}, "
            f"Tenure={ctx.get('tenure','?')} months, "
            f"Internet={ctx.get('InternetService','?')}, "
            f"Monthly charges=${ctx.get('MonthlyCharges','?')}]\n"
        )
        return header + question

    # ── Public query API ──────────────────────────────────────────────────

    def query(
        self,
        question: str,
        customer_context: Optional[dict] = None,
    ) -> dict:
        """
        Retrieve relevant documents and generate a grounded answer (blocking).

        Returns:
            answer    : str          — LLM-generated response
            sources   : list[str]   — unique source filenames cited
            num_docs  : int         — number of retrieved chunks
        """
        if not self.is_ready():
            return {
                "answer":   "Knowledge base is empty. Upload documents to get started.",
                "sources":  [],
                "num_docs": 0,
            }

        enriched = self._enrich(question, customer_context)
        docs     = self._retriever.invoke(enriched)
        context  = self._fmt_context(docs)
        sources  = list({d.metadata.get("source", "unknown") for d in docs})

        chain = (
            {
                "context":  RunnableLambda(lambda _: context),
                "question": RunnablePassthrough(),
            }
            | _RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

        try:
            answer = chain.invoke(enriched)
            return {"answer": answer, "sources": sources, "num_docs": len(docs)}
        except Exception as exc:
            return {
                "answer":   f"⚠️ Query failed: {exc}",
                "sources":  sources,
                "num_docs": len(docs),
            }

    def stream_query(
        self,
        question: str,
        customer_context: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """Generator — yields answer tokens as they arrive from the local LLM."""
        if not self.is_ready():
            yield "Knowledge base is empty. Upload documents to get started."
            return

        enriched = self._enrich(question, customer_context)
        docs     = self._retriever.invoke(enriched)
        context  = self._fmt_context(docs)

        chain = (
            {
                "context":  RunnableLambda(lambda _: context),
                "question": RunnablePassthrough(),
            }
            | _RAG_PROMPT
            | self._llm
        )

        try:
            for chunk in chain.stream(enriched):
                yield chunk.content
        except Exception as exc:
            yield f"\n\n⚠️ Error: {exc}"

    # ── Status helpers ────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._vectorstore is not None and self._retriever is not None

    def get_index_stats(self) -> dict:
        if self._vectorstore is None:
            return {"total_vectors": 0, "ready": False}
        try:
            return {"total_vectors": int(self._vectorstore.index.ntotal), "ready": True}
        except Exception:
            return {"total_vectors": "?", "ready": True}