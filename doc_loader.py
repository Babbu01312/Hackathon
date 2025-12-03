from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# header_template = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#     "Accept-Language": "en-US,en;q=0.5",
#     "Referer": "https://www.google.com/",
#     "DNT": "1",
#     "Connection": "keep-alive",
#     "Upgrade-Insecure-Requests": "1"
# }
docs = []

class document_loader:
    def __init__(self, rag_files=None, rag_links=None):
        self.rag_files = rag_files or []
        self.rag_links = rag_links or []
        if self.rag_files or self.rag_links:
            print(self.rag_files, self.rag_links)
            self._build_vectorstore()

    def _build_vectorstore(self):
        # Load PDFs
        for file_path in self.rag_files:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                loader = UnstructuredExcelLoader(file_path)
                docs.extend(loader.load())
        # Load web links
        for link in self.rag_links:
            loader = WebBaseLoader(
                web_path=link,
                # header_template=header_template,
                requests_per_second=1,
                trust_env=True
            )
            print("*",loader)
            try:
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {link}: {e}")
        print(docs)
        return docs
    
document_loader(rag_links=["https://deepeval.com/guides/guides-ai-agent-evaluation"])
print(docs)
