import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredMarkdownLoader
)

def get_loader_for_file(file_path):
    """Factory to choose the right loader based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path)
    return None

def load_documents(source):
    """
    Loads documents from a file path (str) or Streamlit upload (object).
    Does NOT auto-load profile data; passes strict single source.
    """
    if not source:
        return []

    # CASE 1: Source is a File Path (String)
    if isinstance(source, str):
        if not os.path.exists(source):
            print(f"❌ [Loader] File not found: {source}")
            return []
        
        loader = get_loader_for_file(source)
        if loader:
            try:
                print(f"📄 [Loader] Loading file: {source}")
                return loader.load()
            except Exception as e:
                print(f"❌ [Loader] Error: {e}")
        else:
            print(f"⚠️ [Loader] Unsupported file type: {source}")
        return []

    # CASE 2: Source is a Streamlit UploadedFile Object
    elif hasattr(source, 'name'):
        print(f"📂 [Loader] Processing Streamlit upload: {source.name}")
        suffix = os.path.splitext(source.name)[1]
        
        # Save to temp file because loaders need a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(source.getvalue())
            tmp_path = tmp.name

        docs = []
        try:
            loader = get_loader_for_file(tmp_path)
            if loader:
                docs = loader.load()
                # Fix metadata since temp file loses original name
                for doc in docs:
                    doc.metadata["source"] = source.name
            else:
                print(f"⚠️ [Loader] Unsupported format: {source.name}")
        except Exception as e:
            print(f"❌ [Loader] Streamlit Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return docs

    return []

if __name__ == "__main__":
    # Test with string path
    docs = load_documents("data/MyData.md")
    print(f"Loaded {len(docs)} chunks from file path.")