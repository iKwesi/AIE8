import os
from typing import List, Dict, Any
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PDFFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8", use_pymupdf: bool = True):
        """
        Initialize PDF loader with path and preferred PDF library.
        
        Args:
            path: Path to PDF file or directory containing PDF files
            encoding: Text encoding (for consistency with other loaders)
            use_pymupdf: Whether to use PyMuPDF (fitz) over PyPDF2 if available
        """
        self.documents = []
        self.metadata = []
        self.path = path
        self.encoding = encoding
        self.use_pymupdf = use_pymupdf
        
        # Check available PDF libraries
        if not PDF_AVAILABLE and not PYMUPDF_AVAILABLE:
            raise ImportError(
                "No PDF library available. Please install PyPDF2 or PyMuPDF:\n"
                "pip install PyPDF2\n"
                "or\n"
                "pip install PyMuPDF"
            )

    def load(self):
        """Load PDF files from the specified path."""
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.lower().endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        """Load a single PDF file."""
        if self.use_pymupdf and PYMUPDF_AVAILABLE:
            self._load_file_pymupdf()
        elif PDF_AVAILABLE:
            self._load_file_pypdf2()
        else:
            raise ImportError("No suitable PDF library available")

    def _load_file_pymupdf(self):
        """Load PDF using PyMuPDF (fitz) - generally better text extraction."""
        try:
            doc = fitz.open(self.path)
            full_text = ""
            metadata = {
                "source": self.path,
                "total_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "loader": "PyMuPDF"
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            doc.close()
            self.documents.append(full_text.strip())
            self.metadata.append(metadata)
            
        except Exception as e:
            raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")

    def _load_file_pypdf2(self):
        """Load PDF using PyPDF2 - fallback option."""
        try:
            with open(self.path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                # Extract metadata
                metadata = {
                    "source": self.path,
                    "total_pages": len(pdf_reader.pages),
                    "loader": "PyPDF2"
                }
                
                # Try to get PDF metadata
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get("/Title", ""),
                        "author": pdf_reader.metadata.get("/Author", ""),
                        "subject": pdf_reader.metadata.get("/Subject", ""),
                        "creator": pdf_reader.metadata.get("/Creator", ""),
                        "producer": pdf_reader.metadata.get("/Producer", ""),
                        "creation_date": str(pdf_reader.metadata.get("/CreationDate", "")),
                        "modification_date": str(pdf_reader.metadata.get("/ModDate", ""))
                    })
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                
                self.documents.append(full_text.strip())
                self.metadata.append(metadata)
                
        except Exception as e:
            raise ValueError(f"Error reading PDF file {self.path}: {str(e)}")

    def load_directory(self):
        """Load all PDF files from a directory."""
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    # Temporarily store current path and load individual file
                    original_path = self.path
                    self.path = file_path
                    self.load_file()
                    self.path = original_path

    def load_documents(self):
        """Load documents and return them along with metadata."""
        self.load()
        return self.documents

    def load_documents_with_metadata(self):
        """Load documents and return both documents and metadata."""
        self.load()
        return self.documents, self.metadata


class PDFTextSplitter:
    """
    Enhanced text splitter that preserves PDF-specific metadata during chunking.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_page_info: bool = True,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_page_info = preserve_page_info

    def split(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks while preserving metadata.
        
        Returns:
            List of dictionaries containing 'text' and 'metadata' keys
        """
        chunks = []
        base_metadata = metadata or {}
        
        for i, chunk_start in enumerate(range(0, len(text), self.chunk_size - self.chunk_overlap)):
            chunk_text = text[chunk_start:chunk_start + self.chunk_size]
            
            # Create metadata for this chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_start": chunk_start,
                "chunk_end": min(chunk_start + self.chunk_size, len(text)),
                "chunk_size": len(chunk_text)
            })
            
            # Try to determine which page(s) this chunk spans
            if self.preserve_page_info and "--- Page " in chunk_text:
                page_markers = [line for line in chunk_text.split('\n') if line.startswith('--- Page ')]
                if page_markers:
                    # Extract page numbers
                    pages = []
                    for marker in page_markers:
                        try:
                            page_num = int(marker.split('Page ')[1].split(' ---')[0])
                            pages.append(page_num)
                        except (IndexError, ValueError):
                            pass
                    if pages:
                        chunk_metadata["pages_spanned"] = pages
                        chunk_metadata["primary_page"] = pages[0]
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks

    def split_texts_with_metadata(self, texts: List[str], metadata_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split multiple texts with their corresponding metadata.
        """
        all_chunks = []
        metadata_list = metadata_list or [{}] * len(texts)
        
        for text, metadata in zip(texts, metadata_list):
            chunks = self.split(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks


if __name__ == "__main__":
    # Example usage
    try:
        # Test PDF loading
        loader = PDFFileLoader("sample.pdf")
        documents, metadata = loader.load_documents_with_metadata()
        
        print(f"Loaded {len(documents)} PDF documents")
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            print(f"\nDocument {i+1}:")
            print(f"Source: {meta.get('source', 'Unknown')}")
            print(f"Pages: {meta.get('total_pages', 'Unknown')}")
            print(f"Title: {meta.get('title', 'No title')}")
            print(f"Text preview: {doc[:200]}...")
        
        # Test splitting with metadata
        splitter = PDFTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_texts_with_metadata(documents, metadata)
        
        print(f"\nCreated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"Metadata: {chunk['metadata']}")
            print(f"Text preview: {chunk['text'][:100]}...")
            
    except Exception as e:
        print(f"Error in example: {e}")
        print("Note: This example requires a 'sample.pdf' file to run successfully.")
