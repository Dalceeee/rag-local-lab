from langchain_text_splitters import RecursiveCharacterTextSplitter

class LocalChunker:
    def __init__(self, chunk_size=300, overlap=40):

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def naive_chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
            words = text.split()
            chunks = []
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk = " ".join(words[start:end])
                chunks.append(chunk)
                start += chunk_size - overlap
            return chunks
    
    def chunk_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
        