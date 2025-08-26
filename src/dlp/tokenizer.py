"""SentencePiece tokenizer for HRM-DLP DSL text

Handles tokenization of structured DSL format with proper handling of special tokens.
"""

import os
from typing import List, Optional, Union
import sentencepiece as spm
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration for SentencePiece tokenizer"""
    vocab_size: int = 16000
    model_type: str = "bpe"  # or "unigram"
    character_coverage: float = 0.9995
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    pad_piece: str = "<pad>"
    unk_piece: str = "<unk>"
    bos_piece: str = "<s>"
    eos_piece: str = "</s>"
    
    # DSL-specific control tokens
    control_symbols: List[str] = None
    
    def __post_init__(self):
        if self.control_symbols is None:
            self.control_symbols = [
                "<CHANNEL", "<USER", "<RECIPIENT", "<THREAD", "<META",
                "<SUBJECT>", "</SUBJECT>", "<BODY>", "</BODY>",
                "<ATTACHMENTS>", "</ATTACHMENTS>", "<LINKS>", "</LINKS>",
                # Common roles and channels
                "LEGAL", "FINANCE", "HR", "ENG", "MARKETING", "INTERN",
                "email", "chat", "pr", "upload",
                # Common attributes
                "role=", "dept=", "seniority=", "primary=", "all=",
                "id_hash=", "age_days=", "prior_msgs=",
                "base64=", "homoglyph=", "ts="
            ]


class SimpleTokenizer:
    """Simple fallback tokenizer when SentencePiece is not available"""
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        # Create a simple character-level vocabulary
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Add special tokens
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        for i, token in enumerate(special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        
        # Add basic ASCII characters
        current_id = len(special_tokens)
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            if char not in self.char_to_id and current_id < vocab_size:
                self.char_to_id[char] = current_id
                self.id_to_char[current_id] = char
                current_id += 1
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def encode(self, text: str, max_length: int = None) -> List[int]:
        """Encode text to token IDs"""
        if not isinstance(text, str):
            text = str(text)
        
        # Simple character-level encoding
        ids = [self.bos_token_id]  # Start token
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_token_id))
        ids.append(self.eos_token_id)  # End token
        
        # Truncate or pad
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length-1] + [self.eos_token_id]
            else:
                ids.extend([self.pad_token_id] * (max_length - len(ids)))
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        chars = []
        for id_val in ids:
            if id_val == self.pad_token_id:
                break
            if id_val in [self.bos_token_id, self.eos_token_id]:
                continue
            chars.append(self.id_to_char.get(id_val, "<unk>"))
        return "".join(chars)


class DLPTokenizer:
    """SentencePiece tokenizer for DLP DSL text"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.model_path = model_path
        self.sp_model = spm.SentencePieceProcessor()
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            print(f"Model path {model_path} does not exist. Use train() method to create tokenizer.")
    
    def train(self, input_files: List[str], model_prefix: str):
        """
        Train SentencePiece tokenizer on DSL text files
        
        Args:
            input_files: List of text files containing DSL examples
            model_prefix: Prefix for output model files
        """
        # Create control symbols argument
        control_symbols = ",".join(self.config.control_symbols)
        
        # Training arguments
        train_args = [
            f"--input={','.join(input_files)}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={self.config.vocab_size}",
            f"--model_type={self.config.model_type}",
            f"--character_coverage={self.config.character_coverage}",
            f"--pad_id={self.config.pad_id}",
            f"--unk_id={self.config.unk_id}",
            f"--bos_id={self.config.bos_id}",
            f"--eos_id={self.config.eos_id}",
            f"--pad_piece={self.config.pad_piece}",
            f"--unk_piece={self.config.unk_piece}",
            f"--bos_piece={self.config.bos_piece}",
            f"--eos_piece={self.config.eos_piece}",
            f"--control_symbols={control_symbols}",
            "--normalization_rule_name=nmt_nfkc_cf",
            "--remove_extra_whitespaces=false",  # Preserve DSL structure
            "--max_sentence_length=8192",
        ]
        
        print(f"Training SentencePiece tokenizer with vocab_size={self.config.vocab_size}")
        spm.SentencePieceTrainer.train(" ".join(train_args))
        
        # Load the trained model
        model_path = f"{model_prefix}.model"
        self.load(model_path)
        self.model_path = model_path
        
        print(f"Tokenizer saved to {model_path}")
    
    def load(self, model_path: str):
        """Load existing SentencePiece model"""
        self.sp_model.load(model_path)
        self.model_path = model_path
        print(f"Loaded tokenizer from {model_path}")
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs"""
        if not hasattr(self.sp_model, 'encode_as_ids'):
            raise ValueError("Tokenizer not loaded. Use load() or train() first.")
            
        token_ids = self.sp_model.encode_as_ids(text)
        
        if add_bos:
            token_ids = [self.config.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.config.eos_id]
            
        return token_ids
    
    def decode(self, token_ids: Union[List[int], int]) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.sp_model.decode_ids(token_ids)
    
    def encode_pieces(self, text: str) -> List[str]:
        """Encode text to token pieces (for debugging)"""
        return self.sp_model.encode_as_pieces(text)
    
    def decode_pieces(self, pieces: List[str]) -> str:
        """Decode token pieces to text"""
        return self.sp_model.decode_pieces(pieces)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        if hasattr(self.sp_model, 'get_piece_size'):
            return self.sp_model.get_piece_size()
        return self.config.vocab_size
    
    @property 
    def pad_token_id(self) -> int:
        """Get pad token ID"""
        return self.config.pad_id
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID"""
        return self.config.unk_id
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID"""
        return self.config.bos_id
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID"""
        return self.config.eos_id


def prepare_training_data(jsonl_files: List[str], output_file: str, serializer=None):
    """
    Extract DSL text from JSONL files for tokenizer training
    
    Args:
        jsonl_files: List of JSONL files containing DLP examples
        output_file: Output text file for tokenizer training
        serializer: DSL serializer (optional, will create if None)
    """
    if serializer is None:
        from .dsl import DSLSerializer
        serializer = DSLSerializer()
    
    import json
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for jsonl_file in jsonl_files:
            if not os.path.exists(jsonl_file):
                print(f"Warning: {jsonl_file} not found, skipping")
                continue
                
            print(f"Processing {jsonl_file}")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        data = json.loads(line)
                        
                        # Ensure data is a dictionary
                        if not isinstance(data, dict):
                            print(f"Warning: Line {line_num} is not a dict, skipping")
                            continue
                            
                        # Fix attachments format if needed (like in dataset.py)
                        if "attachments" in data and isinstance(data["attachments"], str):
                            data["attachments"] = []  # Convert single string to empty list for now
                        elif "attachments" in data and isinstance(data["attachments"], list):
                            fixed_attachments = []
                            for att in data["attachments"]:
                                if isinstance(att, str):
                                    fixed_attachments.append({"name": att, "size": 0, "mime": "text/plain"})
                                else:
                                    fixed_attachments.append(att)
                            data["attachments"] = fixed_attachments
                        
                        result = serializer.serialize(data)
                        out_f.write(result.dsl_text + "\n")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Line {line_num} JSON decode error: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Line {line_num} processing error: {e}")
                        continue
    
    print(f"Training data saved to {output_file}")


def create_tokenizer(
    jsonl_files: List[str], 
    model_prefix: str,
    config: Optional[TokenizerConfig] = None
) -> DLPTokenizer:
    """
    Create and train a tokenizer from JSONL files
    
    Args:
        jsonl_files: List of JSONL files containing training data
        model_prefix: Prefix for tokenizer model files
        config: Tokenizer configuration
        
    Returns:
        Trained DLPTokenizer
    """
    # Prepare training data
    training_data_file = f"{model_prefix}_training_data.txt"
    prepare_training_data(jsonl_files, training_data_file)
    
    # Create and train tokenizer
    tokenizer = DLPTokenizer(config=config)
    tokenizer.train([training_data_file], model_prefix)
    
    # Clean up training data file
    if os.path.exists(training_data_file):
        os.remove(training_data_file)
    
    return tokenizer