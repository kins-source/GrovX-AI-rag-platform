import os
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utils.logger import logger

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".markdown", ".csv", ".doc", ".docx",
    ".ppt", ".pptx", ".xls", ".xlsx",
}


def load_document(file_content: bytes, filename: str):
    """
    Loads a document from bytes into Langchain Documents.
    Supports common PDF, text, CSV, Word, PowerPoint, and Excel files.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
        elif ext in {".txt", ".md", ".markdown"}:
            from langchain_community.document_loaders import UnstructuredFileLoader
            loader = UnstructuredFileLoader(temp_path)
            docs = loader.load()
        elif ext == ".csv":
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(temp_path)
            docs = loader.load()
        elif ext in {".doc", ".docx"}:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(temp_path)
            docs = loader.load()
        elif ext in {".ppt", ".pptx"}:
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(temp_path)
            docs = loader.load()
        elif ext == ".xlsx":
            from langchain_core.documents import Document
            from openpyxl import load_workbook

            workbook = load_workbook(temp_path, read_only=True, data_only=True)
            docs = []
            for worksheet in workbook.worksheets:
                rows = [
                    " | ".join("" if value is None else str(value) for value in row)
                    for row in worksheet.iter_rows(values_only=True)
                ]
                content = "\n".join(row for row in rows if row.strip())
                if content:
                    docs.append(Document(page_content=content, metadata={"sheet": worksheet.title}))
            workbook.close()
        elif ext == ".xls":
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(temp_path)
            docs = loader.load()
        else:
            raise ValueError(
                f"Unsupported file type '{ext or 'unknown'}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )
        
        # Add source metadata
        for doc in docs:
            doc.metadata["source"] = filename
            
        logger.info(f"Loaded {len(docs)} pages/sections from {filename}")
        return docs
    finally:
        os.remove(temp_path)
