#!/usr/bin/env python
"""
Text Entity Adapter

Extracts text features with entity linking and sentiment:
- Token embeddings
- Named entity tags
- Sentiment scores per token/phrase
- Discourse markers
"""

import torch
import torch.nn as nn
from typing import Optional, List
import warnings

from ..ingest import ModalityAdapter, ModalityStream


class TextEntityAdapter(ModalityAdapter):
    """
    Text adapter with entity linking and sentiment.

    Extracts:
    - Token embeddings
    - Entity type indicators
    - Token-level sentiment
    - Discourse role markers
    """

    def __init__(
        self,
        output_dim: int = 256,
        vocab_size: int = 50000,
        embed_dim: int = 768,
        extract_entities: bool = True,
        extract_sentiment: bool = True,
        deterministic: bool = True,
    ):
        """
        Initialize text entity adapter.

        Args:
            output_dim: Output feature dimension
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            extract_entities: Extract entity features
            extract_sentiment: Extract sentiment features
            deterministic: Ensure deterministic output
        """
        super().__init__(
            modality_name="text",
            output_dim=output_dim,
            deterministic=deterministic,
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.extract_entities = extract_entities
        self.extract_sentiment = extract_sentiment

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Entity type embeddings (if enabled)
        if extract_entities:
            # Entity types: PERSON, ORG, LOC, MISC, O (other)
            self.entity_embeddings = nn.Embedding(5, 64)
            entity_dim = 64
        else:
            entity_dim = 0

        # Sentiment embeddings (if enabled)
        if extract_sentiment:
            # Sentiment: positive, negative, neutral
            self.sentiment_embeddings = nn.Embedding(3, 32)
            sentiment_dim = 32
        else:
            sentiment_dim = 0

        total_dim = embed_dim + entity_dim + sentiment_dim

        self.proj = nn.Linear(total_dim, output_dim)

    def extract_entities(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract entity type predictions.

        Args:
            token_ids: Token IDs (seq_len,)

        Returns:
            entity_types: (seq_len,) - entity type IDs
        """
        # In full implementation, would use NER model
        # For now, simulate with dummy predictions
        seq_len = token_ids.shape[0]

        # Mostly "O" (other), some random entities
        entity_types = torch.zeros(seq_len, dtype=torch.long)

        # Randomly assign some tokens as entities
        n_entities = int(seq_len * 0.1)
        if n_entities > 0:
            entity_positions = torch.randperm(seq_len)[:n_entities]
            entity_types[entity_positions] = torch.randint(0, 4, (n_entities,))

        return entity_types

    def extract_sentiment(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract token-level sentiment.

        Args:
            token_ids: Token IDs (seq_len,)

        Returns:
            sentiment: (seq_len,) - sentiment type IDs (0=positive, 1=negative, 2=neutral)
        """
        # In full implementation, would use sentiment model
        # For now, simulate with dummy predictions
        seq_len = token_ids.shape[0]

        # Mostly neutral, some positive/negative
        sentiment = torch.full((seq_len,), 2, dtype=torch.long)  # Neutral

        # Randomly assign some tokens as positive/negative
        n_senti = int(seq_len * 0.2)
        if n_senti > 0:
            senti_positions = torch.randperm(seq_len)[:n_senti]
            sentiment[senti_positions] = torch.randint(0, 2, (n_senti,))

        return sentiment

    def forward(
        self,
        token_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> ModalityStream:
        """
        Process text tokens.

        Args:
            token_ids: Token IDs (batch, seq_len)
            timestamps: Optional timestamps (batch, seq_len)

        Returns:
            ModalityStream with text features
        """
        batch_size, seq_len = token_ids.shape

        # Get token embeddings
        token_embeds = self.token_embeddings(token_ids)  # (batch, seq_len, embed_dim)

        features_list = [token_embeds]

        # Extract entities
        if self.extract_entities:
            entity_types = []
            for i in range(batch_size):
                entity_types.append(self.extract_entities(token_ids[i]))

            entity_types_tensor = torch.stack(entity_types, dim=0)  # (batch, seq_len)
            entity_embeds = self.entity_embeddings(entity_types_tensor)  # (batch, seq_len, 64)

            features_list.append(entity_embeds)

        # Extract sentiment
        if self.extract_sentiment:
            sentiments = []
            for i in range(batch_size):
                sentiments.append(self.extract_sentiment(token_ids[i]))

            sentiments_tensor = torch.stack(sentiments, dim=0)  # (batch, seq_len)
            sentiment_embeds = self.sentiment_embeddings(sentiments_tensor)  # (batch, seq_len, 32)

            features_list.append(sentiment_embeds)

        # Concatenate all features
        combined_features = torch.cat(features_list, dim=2)  # (batch, seq_len, total_dim)

        # Project to output_dim
        features_projected = self.proj(combined_features)  # (batch, seq_len, output_dim)

        # Generate timestamps if not provided (word position as time)
        if timestamps is None:
            timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()
            timestamps = timestamps / seq_len  # Normalize to [0, 1]

        return ModalityStream(
            features=features_projected,
            timestamps=timestamps,
            modality="text",
            confidence=1.0,
            metadata={
                "entities_enabled": self.extract_entities,
                "sentiment_enabled": self.extract_sentiment,
            },
        )
