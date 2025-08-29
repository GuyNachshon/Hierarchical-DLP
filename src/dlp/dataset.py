"""Dataset loader for HRM-DLP training data

Handles JSONL format with DLP-specific schema, tokenization, and BIO tag alignment.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import pydantic

from .dsl import DSLSerializer, create_bio_tags, BIO_TAG_TO_ID, NUM_BIO_TAGS
from .tokenizer import DLPTokenizer, SimpleTokenizer, TokenizerConfig


from typing import Union

class DLPExample(pydantic.BaseModel):
    """Schema for DLP training examples"""
    model_config = pydantic.ConfigDict(extra='allow')
    
    channel: str = "email"
    user: Dict[str, Any] = pydantic.Field(default_factory=dict)
    recipients: List[str] = pydantic.Field(default_factory=list)
    thread: Dict[str, Any] = pydantic.Field(default_factory=dict)
    subject: str = ""
    body: str = ""
    attachments: List[Union[str, Dict[str, Any]]] = pydantic.Field(default_factory=list)
    links: Union[List[Union[str, Dict[str, Any]]], Dict[str, Any], None] = pydantic.Field(default_factory=list)
    labels: Dict[str, float] = pydantic.Field(default_factory=dict)
    spans: List[Dict[str, Any]] = pydantic.Field(default_factory=list)
    meta: Dict[str, Any] = pydantic.Field(default_factory=dict)


@dataclass
class DLPDatasetConfig:
    """Configuration for DLP dataset"""
    max_length: int = 1024
    doc_labels: List[str] = None
    pad_token_id: int = 0
    ignore_label_id: int = -100
    
    def __post_init__(self):
        if self.doc_labels is None:
            self.doc_labels = ["sensitivity", "exposure", "context", "obfuscation"]


@dataclass 
class DLPBatch:
    """Batch of DLP training examples"""
    input_ids: torch.Tensor          # [batch_size, seq_len]
    attention_mask: torch.Tensor     # [batch_size, seq_len]
    doc_labels: torch.Tensor         # [batch_size, num_doc_labels]
    bio_labels: torch.Tensor         # [batch_size, seq_len]
    memory_flags: torch.Tensor       # [batch_size, memory_dim] - for future use
    

class DLPDataset(Dataset):
    """PyTorch Dataset for DLP training data"""
    
    def __init__(
        self, 
        jsonl_path: str,
        tokenizer,
        config: DLPDatasetConfig,
        serializer: Optional[DSLSerializer] = None
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.config = config
        self.serializer = serializer or DSLSerializer()
        
        # Load examples
        self.examples = self._load_examples()
        
        print(f"Loaded {len(self.examples)} examples from {jsonl_path}")
    
    def _load_examples(self) -> List[DLPExample]:
        """Load and validate examples from JSONL file"""
        examples = []
        
        if not os.path.exists(self.jsonl_path):
            print(f"Warning: {self.jsonl_path} does not exist, returning empty dataset")
            return examples
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Fix attachments format if they're strings instead of dicts
                    if "attachments" in data and isinstance(data["attachments"], list):
                        fixed_attachments = []
                        for att in data["attachments"]:
                            if isinstance(att, str):
                                # Convert string filename to dict format
                                fixed_attachments.append({
                                    "name": att,
                                    "size": 0,  # Unknown size
                                    "mime": self._guess_mime_type(att)
                                })
                            else:
                                fixed_attachments.append(att)
                        data["attachments"] = fixed_attachments
                    
                    example = DLPExample(**data)
                    examples.append(example)
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        return examples
    
    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type from filename extension"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        mime_map = {
            'pdf': 'application/pdf',
            'doc': 'application/msword',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xls': 'application/vnd.ms-excel',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'ppt': 'application/vnd.ms-powerpoint',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'txt': 'text/plain',
            'csv': 'text/csv',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'ics': 'text/calendar',
            'zip': 'application/zip'
        }
        return mime_map.get(ext, 'application/octet-stream')
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        example = self.examples[idx]
        
        # Convert to dict for serialization
        example_dict = example.model_dump()
        
        # Serialize to DSL format
        serialization_result = self.serializer.serialize(example_dict)
        dsl_text = serialization_result.dsl_text
        
        # Tokenize
        tokens = self.tokenizer.encode(dsl_text)
        
        # Truncate or pad to max_length
        if len(tokens) > self.config.max_length:
            tokens = tokens[:self.config.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Pad if necessary
        while len(tokens) < self.config.max_length:
            tokens.append(self.config.pad_token_id)
            attention_mask.append(0)
        
        # Create BIO tags
        bio_tags = create_bio_tags(dsl_text, serialization_result.span_mappings, self.tokenizer)
        
        # Convert BIO tags to IDs and pad/truncate
        bio_ids = [BIO_TAG_TO_ID.get(tag, 0) for tag in bio_tags]
        if len(bio_ids) > self.config.max_length:
            bio_ids = bio_ids[:self.config.max_length]
        while len(bio_ids) < self.config.max_length:
            bio_ids.append(self.config.ignore_label_id)
        
        # Extract document labels
        doc_label_values = []
        for label_name in self.config.doc_labels:
            value = example_dict.get("labels", {}).get(label_name, 0)
            doc_label_values.append(float(value))
        
        # Create memory flags placeholder (future use)
        memory_flags = torch.zeros(256)  # 256-D memory vector placeholder
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "doc_labels": torch.tensor(doc_label_values, dtype=torch.float),
            "bio_labels": torch.tensor(bio_ids, dtype=torch.long),
            "memory_flags": memory_flags
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> DLPBatch:
        """Collate function for DataLoader"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        doc_labels = torch.stack([item["doc_labels"] for item in batch])
        bio_labels = torch.stack([item["bio_labels"] for item in batch])
        memory_flags = torch.stack([item["memory_flags"] for item in batch])
        
        return DLPBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            doc_labels=doc_labels,
            bio_labels=bio_labels,
            memory_flags=memory_flags
        )


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    config: DLPDatasetConfig,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation DataLoaders"""
    
    train_dataset = DLPDataset(train_path, tokenizer, config)
    val_dataset = DLPDataset(val_path, tokenizer, config)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


class SimpleTokenizer:
    """Simple tokenizer for development/testing"""
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.unk_token_id = 1
        
        # Simple character-based tokenization for now
        # This should be replaced with SentencePiece
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (simple implementation)"""
        # Convert to bytes and map to vocab range
        byte_ids = text.encode('utf-8')
        token_ids = [min(int(b) + 2, self.vocab_size - 1) for b in byte_ids]
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text (simple implementation)"""
        try:
            byte_values = [max(0, min(255, tid - 2)) for tid in token_ids if tid > 1]
            return bytes(byte_values).decode('utf-8', errors='ignore')
        except:
            return ""