import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

class MINDDataLoader:
    def __init__(self, data_dir: str):
        """Initialize the data loader with the path to the MIND dataset directory."""
        self.data_dir = Path(data_dir)
        self.news_df = None
        self.behaviors_df = None
        self.entity_embeddings = None
        self.relation_embeddings = None

    def load_news_data(self) -> pd.DataFrame:
        """Load and process news data from news.tsv."""
        news_path = self.data_dir / 'news.tsv'
        self.news_df = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        )
        return self.news_df

    def load_behaviors_data(self) -> pd.DataFrame:
        """Load and process user behavior data from behaviors.tsv."""
        behaviors_path = self.data_dir / 'behaviors.tsv'
        self.behaviors_df = pd.read_csv(
            behaviors_path,
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )
        return self.behaviors_df

    def load_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """Load entity embeddings from entity_embedding.vec."""
        entity_path = self.data_dir / 'entity_embedding.vec'
        self.entity_embeddings = {}
        with open(entity_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                entity_id = parts[0]
                embedding = np.array([float(x) for x in parts[1:]])
                self.entity_embeddings[entity_id] = embedding
        return self.entity_embeddings

    def load_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """Load relation embeddings from relation_embedding.vec."""
        relation_path = self.data_dir / 'relation_embedding.vec'
        self.relation_embeddings = {}
        with open(relation_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                relation_id = parts[0]
                embedding = np.array([float(x) for x in parts[1:]])
                self.relation_embeddings[relation_id] = embedding
        return self.relation_embeddings

    def process_user_history(self) -> Dict[str, List[str]]:
        """Process user history from behaviors data."""
        if self.behaviors_df is None:
            self.load_behaviors_data()
        
        user_history = {}
        for _, row in self.behaviors_df.iterrows():
            if pd.notna(row['history']):
                user_history[row['user_id']] = row['history'].split()
        return user_history

    def process_impressions(self) -> Dict[str, List[Tuple[str, int]]]:
        """Process impressions data to get user-item interactions with labels."""
        if self.behaviors_df is None:
            self.load_behaviors_data()
        
        user_interactions = {}
        for _, row in self.behaviors_df.iterrows():
            if pd.notna(row['impressions']):
                interactions = []
                for impression in row['impressions'].split():
                    news_id, label = impression.split('-')
                    interactions.append((news_id, int(label)))
                user_interactions[row['user_id']] = interactions
        return user_interactions

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load all data components and return them."""
        news_df = self.load_news_data()
        behaviors_df = self.load_behaviors_data()
        entity_embeddings = self.load_entity_embeddings()
        relation_embeddings = self.load_relation_embeddings()
        
        return news_df, behaviors_df, entity_embeddings, relation_embeddings 