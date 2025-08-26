"""DSL Serializer for HRM-DLP

Converts email/chat/PR content into structured DSL format for model input.
Maintains span offset mapping for BIO tag alignment.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re


@dataclass
class SpanMapping:
    """Maps original text spans to serialized DSL offsets"""
    original_start: int
    original_end: int
    serialized_start: int
    serialized_end: int
    span_type: str


@dataclass 
class SerializationResult:
    """Result of DSL serialization"""
    dsl_text: str
    span_mappings: List[SpanMapping]
    body_start_offset: int  # Where body content starts in DSL
    body_end_offset: int    # Where body content ends in DSL


class DSLSerializer:
    """Serializes structured content into DSL format for HRM-DLP"""
    
    def __init__(self):
        # Special tokens for DSL structure
        self.CHANNEL_TOKEN = "<CHANNEL"
        self.USER_TOKEN = "<USER"
        self.RECIPIENT_TOKEN = "<RECIPIENT"
        self.THREAD_TOKEN = "<THREAD"
        self.SUBJECT_TOKEN = "<SUBJECT>"
        self.SUBJECT_END_TOKEN = "</SUBJECT>"
        self.BODY_TOKEN = "<BODY>"
        self.BODY_END_TOKEN = "</BODY>"
        self.ATTACHMENTS_TOKEN = "<ATTACHMENTS>"
        self.ATTACHMENTS_END_TOKEN = "</ATTACHMENTS>"
        self.LINKS_TOKEN = "<LINKS>"
        self.LINKS_END_TOKEN = "</LINKS>"
        self.META_TOKEN = "<META"
        
    def serialize(self, example: Dict[str, Any]) -> SerializationResult:
        """
        Convert structured example to DSL format
        
        Args:
            example: Dictionary containing channel, user, recipients, subject, body, etc.
            
        Returns:
            SerializationResult with DSL text and span mappings
        """
        dsl_parts = []
        current_offset = 0
        
        # Channel
        channel = example.get("channel", "email")
        channel_part = f"{self.CHANNEL_TOKEN} {channel}>\n"
        dsl_parts.append(channel_part)
        current_offset += len(channel_part)
        
        # User info
        user = example.get("user", {})
        user_attrs = []
        for key in ["role", "dept", "seniority"]:
            if key in user:
                user_attrs.append(f"{key}={user[key]}")
        user_part = f"{self.USER_TOKEN} {' '.join(user_attrs)}>\n"
        dsl_parts.append(user_part)
        current_offset += len(user_part)
        
        # Recipients
        recipients = example.get("recipients", [])
        if recipients:
            primary = recipients[0] if recipients else ""
            all_recipients = ",".join(recipients)
            recipient_part = f"{self.RECIPIENT_TOKEN} primary={primary} all=[{all_recipients}]>\n"
        else:
            recipient_part = f"{self.RECIPIENT_TOKEN}>\n"
        dsl_parts.append(recipient_part)
        current_offset += len(recipient_part)
        
        # Thread info
        thread = example.get("thread", {})
        thread_attrs = []
        for key in ["id_hash", "age_days", "prior_msgs"]:
            if key in thread:
                thread_attrs.append(f"{key}={thread[key]}")
        thread_part = f"{self.THREAD_TOKEN} {' '.join(thread_attrs)}>\n"
        dsl_parts.append(thread_part)
        current_offset += len(thread_part)
        
        # Subject
        subject = example.get("subject", "")
        subject_part = f"{self.SUBJECT_TOKEN}{subject}{self.SUBJECT_END_TOKEN}\n"
        dsl_parts.append(subject_part)
        current_offset += len(subject_part)
        
        # Body (this is where we need to track spans)
        body = example.get("body", "")
        body_start_token = f"{self.BODY_TOKEN}"
        body_end_token = f"{self.BODY_END_TOKEN}\n"
        
        body_start_offset = current_offset + len(body_start_token)
        body_end_offset = body_start_offset + len(body)
        
        body_part = f"{body_start_token}{body}{body_end_token}"
        dsl_parts.append(body_part)
        current_offset += len(body_part)
        
        # Attachments
        attachments = example.get("attachments", [])
        if attachments:
            attachment_strs = []
            for att in attachments:
                name = att.get("name", "")
                size = att.get("size", 0)
                mime = att.get("mime", "")
                size_kb = size // 1024 if size > 0 else 0
                attachment_strs.append(f"{name}|{size_kb}KB;{mime}")
            attachments_content = ",".join(attachment_strs)
            attachments_part = f"{self.ATTACHMENTS_TOKEN}{attachments_content}{self.ATTACHMENTS_END_TOKEN}\n"
        else:
            attachments_part = f"{self.ATTACHMENTS_TOKEN}{self.ATTACHMENTS_END_TOKEN}\n"
        dsl_parts.append(attachments_part)
        current_offset += len(attachments_part)
        
        # Links
        links_raw = example.get("links", [])
        if links_raw:
            link_strs = []
            
            # Handle case where links is a dict instead of list
            if isinstance(links_raw, dict):
                for key, value in links_raw.items():
                    if isinstance(value, str):
                        link_strs.append(f"{value}|{key}")
                    else:
                        link_strs.append(str(value))
            elif isinstance(links_raw, list):
                # Handle list of links (string or dict format)
                for link in links_raw:
                    if isinstance(link, str):
                        link_strs.append(link)
                    elif isinstance(link, dict):
                        url = link.get("url", "")
                        label = link.get("label", "")
                        if label:
                            link_strs.append(f"{url}|{label}")
                        else:
                            link_strs.append(url)
            else:
                # Fallback for other types
                link_strs.append(str(links_raw))
            
            links_content = ",".join(link_strs)
            links_part = f"{self.LINKS_TOKEN}{links_content}{self.LINKS_END_TOKEN}\n"
        else:
            links_part = f"{self.LINKS_TOKEN}{self.LINKS_END_TOKEN}\n"
        dsl_parts.append(links_part)
        current_offset += len(links_part)
        
        # Meta
        meta = example.get("meta", {})
        meta_attrs = []
        for key in ["base64", "homoglyph", "ts"]:
            if key in meta:
                meta_attrs.append(f"{key}={meta[key]}")
        meta_part = f"{self.META_TOKEN} {' '.join(meta_attrs)}>\n"
        dsl_parts.append(meta_part)
        
        dsl_text = "".join(dsl_parts)
        
        # Map original spans to DSL offsets
        span_mappings = self._map_spans(
            example.get("spans", []),
            body,
            body_start_offset
        )
        
        return SerializationResult(
            dsl_text=dsl_text,
            span_mappings=span_mappings,
            body_start_offset=body_start_offset,
            body_end_offset=body_end_offset
        )
    
    def _map_spans(self, original_spans: List[Dict], body: str, body_offset: int) -> List[SpanMapping]:
        """Map original body spans to DSL-serialized positions"""
        mappings = []
        
        for span in original_spans:
            orig_start = span.get("start", 0)
            orig_end = span.get("end", 0)
            span_type = span.get("type", "")
            
            # Validate span is within body bounds
            if orig_start >= 0 and orig_end <= len(body) and orig_start < orig_end:
                # Direct mapping since body text is preserved in DSL
                serialized_start = body_offset + orig_start
                serialized_end = body_offset + orig_end
                
                mappings.append(SpanMapping(
                    original_start=orig_start,
                    original_end=orig_end,
                    serialized_start=serialized_start,
                    serialized_end=serialized_end,
                    span_type=span_type
                ))
        
        return mappings


def create_bio_tags(dsl_text: str, span_mappings: List[SpanMapping], tokenizer) -> List[str]:
    """
    Create BIO tags for tokenized DSL text
    
    Args:
        dsl_text: Serialized DSL string
        span_mappings: List of span mappings from serialization
        tokenizer: Tokenizer with encode/decode methods
        
    Returns:
        List of BIO tags aligned with tokens
    """
    # Tokenize DSL text
    token_ids = tokenizer.encode(dsl_text)
    
    # Create character-to-token mapping
    char_to_token = []
    current_pos = 0
    
    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id])
        token_len = len(token_text)
        
        for _ in range(token_len):
            if current_pos < len(dsl_text):
                char_to_token.append(i)
            current_pos += 1
    
    # Initialize all tags as 'O' (Outside)
    bio_tags = ['O'] * len(token_ids)
    
    # Apply span tags
    for mapping in span_mappings:
        start_char = mapping.serialized_start
        end_char = mapping.serialized_end
        span_type = mapping.span_type
        
        if start_char < len(char_to_token) and end_char <= len(char_to_token):
            start_token = char_to_token[start_char]
            end_token = char_to_token[end_char - 1] if end_char > 0 else start_token
            
            # Apply B- and I- tags
            for token_idx in range(start_token, end_token + 1):
                if token_idx < len(bio_tags):
                    if token_idx == start_token:
                        bio_tags[token_idx] = f"B-{span_type}"
                    else:
                        bio_tags[token_idx] = f"I-{span_type}"
    
    return bio_tags


# Span type mapping for BIO tags
SPAN_TYPES = [
    "EMAIL", "PHONE", "PAN", "SSN", "SECRET", "DBURI", 
    "NDA", "MATTER", "NAME", "ADDR"
]

BIO_TAG_TO_ID = {"O": 0}
for i, span_type in enumerate(SPAN_TYPES):
    BIO_TAG_TO_ID[f"B-{span_type}"] = 2 * i + 1
    BIO_TAG_TO_ID[f"I-{span_type}"] = 2 * i + 2

ID_TO_BIO_TAG = {v: k for k, v in BIO_TAG_TO_ID.items()}
NUM_BIO_TAGS = len(BIO_TAG_TO_ID)