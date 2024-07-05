from typing import List, Optional
from abc import ABC, abstractmethod

DEFAULT_CHUNK_LOCATION = "chunks"


class BaseParser(ABC):
    def __init__(self,
                 save_chunks: bool = False,
                 save_location: Optional[str] = None,
                 **kwargs):
        self.save_chunks = save_chunks
        self.save_location = save_location or DEFAULT_CHUNK_LOCATION

    def parse_file(self, file_path: str, **kwargs):
        chunks = self._parse_file(file_path, **kwargs)
        if self.save_chunks:
            for i, chunk in enumerate(chunks):
                with open(f"{self.save_location}/chunk_{i}.txt", 'w') as f:
                    f.write(chunk)

    @abstractmethod
    def _parse_file(self, file_path: str, **kwargs) -> List[str]:
        ...