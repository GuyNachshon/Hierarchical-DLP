"""
DLP Format Converter for HRM-DLP Training Data

Converts agentic data generator outputs into proper HRM-DLP training format
with span extraction, document labeling, and format validation.
"""

import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class DLPExample:
    """HRM-DLP training example structure."""
    channel: str
    user: Dict[str, str]
    recipients: List[str]
    thread: Dict[str, Any]
    subject: str
    body: str
    attachments: List[Dict[str, Any]]
    links: List[str]
    labels: Dict[str, int]
    spans: List[Dict[str, Any]]
    meta: Dict[str, Any]


class DLPSpanExtractor:
    """Extract PII and sensitive spans from text content."""

    def __init__(self):
        # BIO tag mapping for HRM-DLP
        self.bio_tags = {
            "O": 0,  # Outside
            "B-EMAIL": 1, "I-EMAIL": 2,
            "B-PHONE": 3, "I-PHONE": 4,
            "B-PAN": 5, "I-PAN": 6,
            "B-SSN": 7, "I-SSN": 8,
            "B-NAME": 9, "I-NAME": 10,
            "B-SECRET": 11, "I-SECRET": 12,
            "B-DBURI": 13, "I-DBURI": 14,
            "B-NDA": 15, "I-NDA": 16,
            "B-MATTER": 17, "I-MATTER": 18,
            "B-ACCOUNT": 19, "I-ACCOUNT": 20
        }

        # Span extraction patterns
        self.patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'(?:\+\d{1,3}[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            "PAN": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            "SSN": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "SECRET": r'\b(?:AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{48}|xoxp-[a-z0-9\-]{72}|ghp_[A-Za-z0-9]{36})\b',
            "DBURI": r'\b(?:postgresql|mysql|mongodb|redis|sqlite)://[^\s]+\b',
            "NDA": r'\b(?:NDA|Non-Disclosure|Confidentiality Agreement)\b',
            "MATTER": r'\b(?:Matter [A-Z]?#?\d+[-\d]*|Case #\d+)\b',
            "NAME": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "ACCOUNT": r'\b(?:Account|Acct) #?(\d{4,})\b'
        }

    def extract_spans(self, text: str) -> List[Dict[str, Any]]:
        """Extract all spans from text with BIO tagging."""
        spans = []

        for span_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Validate certain patterns
                if span_type == "PAN" and not self._luhn_check(match.group()):
                    continue

                if span_type == "NAME" and self._is_false_positive_name(match.group()):
                    continue

                spans.append({
                    "type": span_type,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group()
                })

        # Sort by position and remove overlaps
        spans = sorted(spans, key=lambda x: x["start"])
        spans = self._remove_overlapping_spans(spans)

        return spans

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        digits = [int(d) for d in card_number if d.isdigit()]
        total = 0
        reverse_digits = digits[::-1]

        for i, digit in enumerate(reverse_digits):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit

        return total % 10 == 0

    def _is_false_positive_name(self, name: str) -> bool:
        """Check if name is likely a false positive."""
        false_positives = [
            "New York", "Los Angeles", "United States", "Data Loss",
            "Legal Matter", "Account Number", "Credit Card", "Social Security"
        ]
        return any(fp.lower() in name.lower() for fp in false_positives)

    def _remove_overlapping_spans(self, spans: List[Dict]) -> List[Dict]:
        """Remove overlapping spans, keeping the longer one."""
        if not spans:
            return []

        result = [spans[0]]

        for current in spans[1:]:
            last = result[-1]

            # Check for overlap
            if current["start"] < last["end"]:
                # Keep the longer span
                if (current["end"] - current["start"]) > (last["end"] - last["start"]):
                    result[-1] = current
                # If same length, prefer certain types
                elif (current["end"] - current["start"]) == (last["end"] - last["start"]):
                    priority = {"PAN": 5, "SSN": 4, "EMAIL": 3, "PHONE": 2, "NAME": 1}
                    if priority.get(current["type"], 0) > priority.get(last["type"], 0):
                        result[-1] = current
            else:
                result.append(current)

        return result


class BusinessContextAnalyzer:
    """Analyze email content to determine business context and attachment sensitivity patterns."""
    
    def __init__(self):
        # Business domain patterns for context detection
        self.domain_patterns = {
            "payroll": {
                "keywords": ["payroll", "salary", "wages", "employee", "benefits", "hr", "human resources", 
                           "compensation", "paystub", "w2", "w-2", "tax", "deduction"],
                "sensitivity_indicators": ["contains_pii", "salary_information", "personal_data", "tax_data"]
            },
            "legal": {
                "keywords": ["nda", "non-disclosure", "legal", "attorney", "counsel", "contract", "agreement",
                           "confidential", "privileged", "litigation", "matter", "case", "settlement"],
                "sensitivity_indicators": ["attorney_client_privilege", "legal", "confidential_terms", "attorney_work_product"]
            },
            "financial": {
                "keywords": ["financial", "revenue", "profit", "budget", "accounting", "invoice", "payment",
                           "banking", "account", "transaction", "credit", "debit", "balance", "audit"],
                "sensitivity_indicators": ["financial_data", "revenue_projections", "banking_data", "account_numbers"]
            },
            "security": {
                "keywords": ["security", "password", "credential", "api", "token", "access", "authentication",
                           "incident", "breach", "vulnerability", "keys", "cert", "certificate"],
                "sensitivity_indicators": ["contains_credentials", "api_keys", "production_access", "security_incident"]
            },
            "customer": {
                "keywords": ["customer", "client", "support", "ticket", "case", "complaint", "feedback",
                           "personal", "contact", "phone", "address", "account"],
                "sensitivity_indicators": ["contains_pii", "customer_data", "contact_information"]
            }
        }
        
        # File extension to content type mapping
        self.file_types = {
            ".xlsx": {"mime": "application/vnd.ms-excel", "likely_content": "spreadsheet_data"},
            ".xls": {"mime": "application/vnd.ms-excel", "likely_content": "spreadsheet_data"},
            ".pdf": {"mime": "application/pdf", "likely_content": "document"},
            ".docx": {"mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "likely_content": "document"},
            ".doc": {"mime": "application/msword", "likely_content": "document"},
            ".txt": {"mime": "text/plain", "likely_content": "text_data"},
            ".csv": {"mime": "text/csv", "likely_content": "data_export"},
            ".zip": {"mime": "application/zip", "likely_content": "archive"},
            ".json": {"mime": "application/json", "likely_content": "data_export"}
        }
    
    def analyze_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email content to determine business context."""
        subject = example.get("subject", "").lower()
        body = example.get("body", "").lower()
        content = f"{subject} {body}"
        
        # Determine primary business domain
        domain_scores = {}
        for domain, config in self.domain_patterns.items():
            score = sum(1 for keyword in config["keywords"] if keyword in content)
            if score > 0:
                domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
        
        # Analyze recipient risk
        recipients = example.get("recipients", [])
        external_personal_domains = {"gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "proton.me", "aol.com", "icloud.com"}
        
        recipient_risk = "low"
        has_external_personal = any(
            any(domain in recipient.lower() for domain in external_personal_domains)
            for recipient in recipients
        )
        
        # Check for competitor or high-risk external domains  
        competitor_indicators = ["competitor", "rival", "vendor", "partner"]
        has_competitor = any(
            any(indicator in recipient.lower() for indicator in competitor_indicators)
            for recipient in recipients
        )
        
        if has_external_personal:
            recipient_risk = "high"
        elif has_competitor:
            recipient_risk = "medium"
        elif any("@" in r and not r.endswith(".internal") and not r.startswith("#") for r in recipients):
            recipient_risk = "medium"
        
        return {
            "primary_domain": primary_domain,
            "domain_confidence": domain_scores.get(primary_domain, 0),
            "recipient_risk": recipient_risk,
            "content_analysis": content
        }
    
    def generate_attachment_metadata(self, filename: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rich attachment metadata based on filename and business context."""
        # Get file extension and basic info
        name = filename.lower()
        ext = next((ext for ext in self.file_types.keys() if name.endswith(ext)), "")
        file_info = self.file_types.get(ext, {"mime": "application/octet-stream", "likely_content": "unknown"})
        
        # Base attachment metadata
        attachment = {
            "name": filename,
            "size": 0,  # We don't have real size data
            "mime": file_info["mime"]
        }
        
        # Generate content summary based on context and filename
        primary_domain = context["primary_domain"]
        
        if primary_domain == "payroll":
            if "payroll" in name or "salary" in name or "employee" in name:
                attachment["content_summary"] = f"Payroll document containing employee compensation and personal data"
                attachment["sensitivity_indicators"] = ["contains_pii", "salary_information", "personal_data"]
            else:
                attachment["content_summary"] = f"HR-related document with potential employee information"
                attachment["sensitivity_indicators"] = ["personal_data"]
        
        elif primary_domain == "legal":
            if "nda" in name or "agreement" in name or "contract" in name:
                attachment["content_summary"] = f"Legal agreement or contract with confidential terms"
                attachment["sensitivity_indicators"] = ["attorney_client_privilege", "confidential_terms"]
            else:
                attachment["content_summary"] = f"Legal document with potential privileged information"
                attachment["sensitivity_indicators"] = ["attorney_work_product", "legal"]
        
        elif primary_domain == "financial":
            if "report" in name or "financial" in name or "revenue" in name:
                attachment["content_summary"] = f"Financial report containing revenue and business metrics"
                attachment["sensitivity_indicators"] = ["financial_data", "revenue_projections"]
            elif "payment" in name or "invoice" in name:
                attachment["content_summary"] = f"Payment document with account and transaction details"
                attachment["sensitivity_indicators"] = ["banking_data", "account_numbers"]
            else:
                attachment["content_summary"] = f"Financial document with business data"
                attachment["sensitivity_indicators"] = ["financial_data"]
        
        elif primary_domain == "security":
            attachment["content_summary"] = f"Security-related document with potential credentials or access information"
            attachment["sensitivity_indicators"] = ["contains_credentials", "security_incident"]
        
        elif primary_domain == "customer":
            attachment["content_summary"] = f"Customer-related document with personal or contact information"
            attachment["sensitivity_indicators"] = ["contains_pii", "customer_data"]
        
        else:
            # General business document
            attachment["content_summary"] = f"Business document"
            attachment["sensitivity_indicators"] = []
            
            # Add sensitivity based on file type and content clues
            if ext in [".xlsx", ".csv"] and ("data" in name or "export" in name):
                attachment["sensitivity_indicators"].append("data_export")
            if "confidential" in name or "sensitive" in name:
                attachment["sensitivity_indicators"].append("confidential")
        
        # Enhance sensitivity based on recipient risk
        if context["recipient_risk"] == "high":
            if "high_exposure_risk" not in attachment["sensitivity_indicators"]:
                attachment["sensitivity_indicators"].append("high_exposure_risk")
        
        return attachment


class DLPLabelGenerator:
    """Generate document-level labels based on content and context."""

    def generate_labels(self, example: Dict[str, Any], spans: List[Dict]) -> Dict[str, int]:
        """Generate multi-task labels for document classification."""
        labels = {"sensitivity": 0, "exposure": 0, "context": 0, "obfuscation": 0}

        body = example.get("body", "")
        recipients = example.get("recipients", [])
        user = example.get("user", {})
        attachments = example.get("attachments", [])

        # Sensitivity: Based on span types and attachment indicators
        high_sensitivity_spans = {"PAN", "SSN", "SECRET", "DBURI", "ACCOUNT"}
        medium_sensitivity_spans = {"EMAIL", "PHONE", "NAME"}

        high_sens_count = sum(1 for span in spans if span["type"] in high_sensitivity_spans)
        medium_sens_count = sum(1 for span in spans if span["type"] in medium_sensitivity_spans)

        # Check attachment sensitivity indicators
        attachment_sensitive = any(
            any(indicator in att.get("sensitivity_indicators", [])
                for indicator in ["contains_ssn", "contains_pii", "financial_data", "medical_information"])
            for att in attachments
        )

        if high_sens_count > 0 or attachment_sensitive:
            labels["sensitivity"] = 1
        elif medium_sens_count >= 2:
            labels["sensitivity"] = 1

        # Exposure: Based on recipient patterns
        external_personal_domains = {
            "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
            "proton.me", "aol.com", "icloud.com"
        }

        has_external_personal = any(
            any(domain in recipient.lower() for domain in external_personal_domains)
            for recipient in recipients
        )

        # Check for suspicious recipient patterns
        company_domain = self._extract_company_domain(example.get("user", {}).get("email", ""))
        has_external_business = any(
            recipient.split("@")[-1].lower() not in [company_domain, ""]
            and not any(domain in recipient.lower() for domain in external_personal_domains)
            for recipient in recipients
        )

        if has_external_personal:
            labels["exposure"] = 1
        elif has_external_business and labels["sensitivity"] == 1:
            labels["exposure"] = 1

        # Context: Based on business legitimacy
        legal_indicators = any(span["type"] in {"NDA", "MATTER"} for span in spans)
        legal_role = user.get("role") in ["LEGAL", "COMPLIANCE"]
        has_legal_attachments = any(
            "legal" in att.get("content_summary", "").lower() or
            "attorney" in att.get("content_summary", "").lower()
            for att in attachments
        )

        finance_role = user.get("role") in ["FINANCE", "ACCOUNTING"]
        has_finance_context = any(
            term in body.lower() for term in ["payment", "invoice", "bank", "financial"]
        )

        if (legal_indicators and legal_role) or has_legal_attachments:
            labels["context"] = 1
        elif finance_role and has_finance_context and not has_external_personal:
            labels["context"] = 1

        # Obfuscation: Based on content patterns
        obfuscation_patterns = [
            r"base64", r"encoded", r"\*{3,}", r"[A-Za-z0-9+/]{20,}={0,2}",
            r"&#\d+;", r"%[0-9A-Fa-f]{2}"
        ]

        has_obfuscation = any(
            re.search(pattern, body, re.IGNORECASE) for pattern in obfuscation_patterns
        )

        if has_obfuscation:
            labels["obfuscation"] = 1

        return labels

    def _extract_company_domain(self, email: str) -> str:
        """Extract company domain from email."""
        if "@" in email:
            return email.split("@")[-1].lower()
        return ""


class DLPFormatConverter:
    """Convert agentic generator outputs to HRM-DLP training format."""

    def __init__(self):
        self.span_extractor = DLPSpanExtractor()
        self.label_generator = DLPLabelGenerator()
        self.context_analyzer = BusinessContextAnalyzer()

    def convert_example(self, raw_example: Dict[str, Any]) -> Optional[DLPExample]:
        """Convert raw agent output to HRM-DLP format."""
        try:
            # Enhanced validation and debugging for input types
            if not isinstance(raw_example, dict):
                # Better debugging information
                if isinstance(raw_example, str):
                    print(f"⚠️  Skipping string example (length: {len(raw_example)}): {raw_example[:100]}...")
                else:
                    print(f"⚠️  Skipping non-dict example: {type(raw_example)} - {raw_example}")
                return None

            # Validate required fields
            if not raw_example.get("body") and not raw_example.get("subject"):
                print(f"⚠️  Skipping example with no body or subject content")
                return None

            # Extract spans from body text
            body = raw_example.get("body", "")
            spans = self.span_extractor.extract_spans(body)

            # Analyze business context for intelligent attachment processing
            context = self.context_analyzer.analyze_context(raw_example)
            
            # Process attachments with contextual enrichment
            attachments = self._process_attachments(raw_example.get("attachments", []), context)
            
            # Create a temporary example with processed attachments for label generation
            temp_example = raw_example.copy()
            temp_example["attachments"] = attachments
            
            # Generate document labels
            labels = self.label_generator.generate_labels(temp_example, spans)

            # Generate thread metadata
            thread = self._process_thread_metadata(raw_example.get("thread", {}))


            # Create DLP example
            dlp_example = DLPExample(
                channel=raw_example.get("channel", "email"),
                user=raw_example.get("user", {}),
                recipients=raw_example.get("recipients", []),
                thread=thread,
                subject=raw_example.get("subject", ""),
                body=body,
                attachments=attachments,
                links=raw_example.get("links", []),
                labels=labels,
                spans=spans,
                meta={
                    "agent": self._safe_get_agent(raw_example),
                    "context_summary": raw_example.get("context_summary", ""),
                    "generated_at": datetime.now().isoformat(),
                    "converter_version": "1.0"
                }
            )

            return dlp_example

        except Exception as e:
            print(f"❌ Failed to convert example: {e.with_traceback(None)}")
            return None

    def _safe_get_agent(self, raw_example: Dict[str, Any]) -> str:
        """Safely extract agent information from various metadata formats."""
        # Try _metadata first (newer format)
        metadata = raw_example.get("_metadata", {})
        if isinstance(metadata, dict) and "agent_type" in metadata:
            return metadata["agent_type"]

        # Try meta (older format)
        meta = raw_example.get("meta", {})
        if isinstance(meta, dict) and "agent" in meta:
            return meta["agent"]

        # Default fallback
        return "unknown"

    def _process_attachments(self, raw_attachments: List[Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process attachment data with contextual enrichment."""
        processed = []

        for att in raw_attachments:
            if isinstance(att, str):
                # Simple filename - enhance with contextual analysis
                rich_metadata = self.context_analyzer.generate_attachment_metadata(att, context)
                processed.append(rich_metadata)
            elif isinstance(att, dict):
                # Already rich attachment metadata - preserve as-is
                processed.append({
                    "name": att.get("name", "unknown.txt"),
                    "size": att.get("size", 0),
                    "mime": att.get("mime_type", "application/octet-stream"),
                    "content_summary": att.get("content_summary", ""),
                    "sensitivity_indicators": att.get("sensitivity_indicators", [])
                })
            else:
                # Unexpected type - create minimal attachment
                processed.append({
                    "name": str(att),
                    "size": 0,
                    "mime": "application/octet-stream",
                    "content_summary": "Unknown attachment type",
                    "sensitivity_indicators": []
                })

        return processed

    def _process_thread_metadata(self, raw_thread: Dict[str, Any]) -> Dict[str, Any]:
        """Process thread metadata for conversation context."""
        if not raw_thread:
            # Generate new thread metadata
            return {
                "id_hash": self._generate_thread_id(),
                "age_days": 0,
                "prior_msgs": 0
            }

        return {
            "id_hash": raw_thread.get("id_hash", self._generate_thread_id()),
            "age_days": raw_thread.get("age_days", 0),
            "prior_msgs": raw_thread.get("prior_msgs", 0)
        }

    def _generate_thread_id(self) -> str:
        """Generate unique thread ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]

    def convert_batch(self, raw_examples: List[Dict[str, Any]]) -> List[DLPExample]:
        """Convert batch of raw examples with enhanced error tracking."""
        converted = []
        error_stats = {
            "non_dict": 0,
            "empty_content": 0,
            "conversion_error": 0,
            "total_processed": 0
        }

        for i, raw_example in enumerate(raw_examples):
            error_stats["total_processed"] += 1

            # Track different types of errors
            if not isinstance(raw_example, dict):
                error_stats["non_dict"] += 1
                continue

            if not raw_example.get("body") and not raw_example.get("subject"):
                error_stats["empty_content"] += 1
                continue

            try:
                dlp_example = self.convert_example(raw_example)
                if dlp_example:
                    converted.append(dlp_example)
            except Exception as e:
                error_stats["conversion_error"] += 1
                if error_stats["conversion_error"] <= 5:  # Only print first 5 errors
                    print(f"❌ Conversion error on example {i}: {e}")

        # Print summary statistics (only once per batch)
        success_rate = len(converted) / error_stats["total_processed"] * 100 if error_stats["total_processed"] > 0 else 0
        print(f"   📊 Conversion summary: {len(converted)}/{error_stats['total_processed']} successful ({success_rate:.1f}%)")

        if error_stats["non_dict"] > 0:
            print(f"      • Non-dict objects: {error_stats['non_dict']}")
        if error_stats["empty_content"] > 0:
            print(f"      • Empty content: {error_stats['empty_content']}")
        if error_stats["conversion_error"] > 0:
            print(f"      • Conversion errors: {error_stats['conversion_error']}")

        return converted

    def to_jsonl(self, examples: List[DLPExample]) -> List[str]:
        """Convert DLP examples to JSONL format."""
        jsonl_lines = []

        for example in examples:
            # Convert to dictionary format expected by HRM-DLP training
            example_dict = {
                "channel": example.channel,
                "user": example.user,
                "recipients": example.recipients,
                "thread": example.thread,
                "subject": example.subject,
                "body": example.body,
                "attachments": example.attachments,
                "links": example.links,
                "labels": example.labels,
                "spans": example.spans,
                "meta": example.meta
            }

            jsonl_lines.append(json.dumps(example_dict))

        return jsonl_lines


# Quality validation functions
def validate_example_quality(example: DLPExample) -> Tuple[bool, List[str]]:
    """Validate example meets quality requirements."""
    issues = []

    # Check required fields - be more lenient
    if not example.body or len(example.body) < 10:  # Reduced from 50 to 10
        issues.append("Body too short or empty")

    # Subject is not strictly required for all communication types
    # if not example.subject:
    #     issues.append("Missing subject")

    if not example.recipients:
        issues.append("No recipients")

    # Check span requirements - be more lenient
    # Don't require spans for all examples - some legitimate communications have no PII
    # if len(example.spans) == 0:
    #     issues.append("No spans extracted")
    if len(example.spans) > 20:  # Increased from 10 to 20
        issues.append("Too many spans (>20)")

    # Check label consistency - be more lenient
    # Allow sensitive content without spans - it might be contextually sensitive
    # if example.labels.get("sensitivity", 0) == 1 and len(example.spans) == 0:
    #     issues.append("Sensitive label but no spans")

    # Check realistic content
    if "lorem ipsum" in example.body.lower():
        issues.append("Contains placeholder text")

    return len(issues) == 0, issues
