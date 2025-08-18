from abc import ABC, abstractmethod
from typing import Dict, List

class FileOrganizer(ABC):
    @abstractmethod
    def organize_by_clusters(self, clusters: List[Dict[str, List[str]]]) -> None:
        pass