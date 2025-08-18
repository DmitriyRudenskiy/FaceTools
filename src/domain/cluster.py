from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Cluster:
    """Группа похожих лиц"""

    id: int
    size: int
    representative: str
    representative_path: str
    members: List[str]
    members_paths: List[str]
    average_similarity: float


@dataclass
class ClusteringResult:
    """Результат кластеризации"""

    timestamp: str
    total_clusters: int
    unrecognized_count: int
    clusters: List[Cluster]
    unrecognized_images: List[Dict[str, Any]]
