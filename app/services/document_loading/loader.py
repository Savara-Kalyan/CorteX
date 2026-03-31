from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader


_loader= {
    "*.pdf": PyMuPDFLoader,
    "*.txt": TextLoader,
    "*.md": UnstructuredMarkdownLoader
}


