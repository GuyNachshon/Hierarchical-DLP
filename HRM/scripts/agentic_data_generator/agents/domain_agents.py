"""
Domain-specific agents for generating realistic business communications.
"""

from typing import Optional, Dict
from .base_agent import BaseLLMAgent, GenerationRequest, GeneratedExample


class LegalAgent(BaseLLMAgent):
    """Legal domain specialist - NDAs, compliance, legal matters."""

    def __init__(self, config):
        super().__init__(config, "Legal")

    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        prompt = f"""Generate a realistic legal communication for HRM-DLP training.

SCENARIO: {request.scenario_context}
RISK LEVEL: {request.risk_level}
TARGET SPANS: {request.target_spans}

Create a detailed legal communication with the following requirements:
1. Realistic business context (law firm, compliance, contracts)
2. Natural inclusion of sensitive information
3. Appropriate legal terminology and processes
4. CRITICAL: Generate complete attachment metadata including realistic file sizes, detailed content summaries, and contextual sensitivity indicators
5. Context summary for multi-task learning

ATTACHMENT GENERATION REQUIREMENTS - CRITICAL:
- Include 1-3 attachments that fit the legal scenario
- Generate realistic file sizes (PDFs: 50KB-2MB, Word docs: 20KB-500KB, Excel: 100KB-1MB)
- Write detailed content summaries that describe what sensitive information the file contains
- Add 2-4 sensitivity indicators per attachment from: [contains_client_pii, attorney_client_privilege, attorney_work_product, litigation_strategy, confidential_terms, nda_content, case_details, settlement_terms, legal_opinions]
- **MANDATORY**: Each attachment MUST be a complete object with name, size, mime_type, content_summary, and sensitivity_indicators
- **FAILURE TO PROVIDE RICH ATTACHMENT METADATA WILL RESULT IN REJECTION**

RISK PATTERNS:
- AUTHORIZED: Legal reviews with external counsel (lawfirm.com domains)
- AUTHORIZED: Matter coordination with approved co-counsel
- VIOLATION: Legal documents to personal emails (gmail, outlook, etc.)
- VIOLATION: Attorney-client privileged content to unauthorized recipients
- OBFUSCATED: Hidden legal terms in attachments or encoded content

Generate JSON with this EXACT structure:
{{
  "channel": "email",
  "user": {{
    "role": "LEGAL", 
    "dept": "LEGAL", 
    "seniority": "SENIOR"
  }},
  "recipients": ["counsel@smithlegal.com", "associate@joneslawfirm.com"],
  "subject": "Matter #2024-156: NDA Review and Coordination",
  "body": "Comprehensive legal communication with naturally embedded sensitive information like case numbers, client names, legal document references, potential PII, etc. Make it feel authentic with proper legal language, case citations, matter numbers, and business context.",
  "attachments": [
    {{
      "name": "nda_draft_v3.pdf",
      "size": 245760,
      "mime_type": "application/pdf",
      "content_summary": "Non-disclosure agreement draft containing client information and confidential terms",
      "sensitivity_indicators": ["contains_client_pii", "attorney_client_privilege"]
    }},
    {{
      "name": "matter_summary.docx", 
      "size": 89600,
      "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "content_summary": "Legal matter summary with case details and strategy notes",
      "sensitivity_indicators": ["attorney_work_product", "litigation_strategy"]
    }}
  ],
  "links": ["https://legal-portal.smithlaw.com/matter/2024-156"],
  "context_summary": "Legal matter coordination regarding NDA review for client acquisition. Discussion of sensitive contract terms and attorney-client privileged information.",
  "thread": {{
    "id_hash": "leg_thread_001",
    "age_days": 3,
    "prior_msgs": 2
  }}
}}

CRITICAL: Embed realistic sensitive information naturally in the body text including:
- Legal matter numbers (e.g., "Matter #2024-156") 
- Client names and company information
- Attorney names and bar numbers
- Case citations and legal references
- Contract amounts and terms
- Confidential legal strategies"""

        system_prompt = "You are an expert legal professional generating realistic legal communications."

        response = await self._generate_with_llm(prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_legal_example(data):
                self.generation_stats["successful"] += 1
                return GeneratedExample(
                    channel=data.get("channel", "email"),
                    user=data.get("user", {}),
                    recipients=data.get("recipients", []),
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    attachments=data.get("attachments", []),
                    links=data.get("links", []),
                    thread=data.get("thread"),
                    labels=data.get("labels"),
                    spans=data.get("spans"),
                    meta={"agent": "legal", **data.get("meta", {})}
                )

        self.generation_stats["failed"] += 1
        return None

    def _validate_legal_example(self, data: Dict) -> bool:
        """Validate legal-specific requirements."""
        body = data.get("body", "")
        legal_terms = ["matter", "nda", "agreement", "confidential", "privilege", "counsel"]
        has_legal_context = any(term in body.lower() for term in legal_terms)

        user = data.get("user", {})
        is_legal_role = user.get("role") in ["LEGAL", "COMPLIANCE", "PRIVACY"]

        # Validate attachment structure
        attachments = data.get("attachments", [])
        if not self._validate_attachment_schema(attachments):
            return False
            
        # Validate legal-specific sensitivity indicators
        legal_indicators = ["contains_client_pii", "attorney_client_privilege", "attorney_work_product", 
                          "litigation_strategy", "confidential_terms", "nda_content", "case_details", 
                          "settlement_terms", "legal_opinions"]
        if not self._validate_sensitivity_indicators(attachments, legal_indicators):
            return False

        return has_legal_context and is_legal_role

    def _validate_attachment_schema(self, attachments: list) -> bool:
        """Validate that all attachments have proper schema structure."""
        if not attachments:
            return True  # Empty attachments list is valid
            
        for attachment in attachments:
            if not isinstance(attachment, dict):
                return False
                
            required_fields = ["name", "size", "mime_type", "content_summary", "sensitivity_indicators"]
            for field in required_fields:
                if field not in attachment:
                    return False
                    
            # Validate field types
            if not isinstance(attachment["name"], str) or not attachment["name"]:
                return False
            if not isinstance(attachment["size"], int) or attachment["size"] <= 0:
                return False
            if not isinstance(attachment["mime_type"], str) or not attachment["mime_type"]:
                return False
            if not isinstance(attachment["content_summary"], str) or not attachment["content_summary"]:
                return False
            if not isinstance(attachment["sensitivity_indicators"], list) or not attachment["sensitivity_indicators"]:
                return False
                
        return True

    def _validate_sensitivity_indicators(self, attachments: list, valid_indicators: list) -> bool:
        """Validate that attachments use appropriate sensitivity indicators for the domain."""
        for attachment in attachments:
            if isinstance(attachment, dict):
                indicators = attachment.get("sensitivity_indicators", [])
                if not any(indicator in valid_indicators for indicator in indicators):
                    return False
        return True


class FinanceAgent(BaseLLMAgent):
    """Finance domain specialist - Payments, financial data, banking."""

    def __init__(self, config):
        super().__init__(config, "Finance")

    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        prompt = f"""Generate realistic finance communication for HRM-DLP training.

SCENARIO: {request.scenario_context}
RISK LEVEL: {request.risk_level}
TARGET SPANS: {request.target_spans}

Create a detailed financial communication with the following requirements:
1. Realistic business context (banking, payments, invoicing, financial reporting)
2. Natural inclusion of financial sensitive information
3. Appropriate financial terminology and processes
4. CRITICAL: Generate complete attachment metadata including realistic file sizes, detailed content summaries, and contextual sensitivity indicators
5. Context summary for multi-task learning

ATTACHMENT GENERATION REQUIREMENTS - CRITICAL:
- Include 1-3 attachments that fit the financial scenario
- Generate realistic file sizes (Excel: 100KB-2MB, PDFs: 50KB-1MB, CSV: 10KB-500KB)
- Write detailed content summaries that describe what financial data the file contains
- Add 2-4 sensitivity indicators per attachment from: [contains_banking_data, payment_credentials, account_numbers, financial_data, revenue_projections, tax_information, credit_card_data, wire_transfer_details, banking_credentials]
- **MANDATORY**: Each attachment MUST be a complete object with name, size, mime_type, content_summary, and sensitivity_indicators
- **FAILURE TO PROVIDE RICH ATTACHMENT METADATA WILL RESULT IN REJECTION**

RISK PATTERNS:
- AUTHORIZED: Payment processing with established bank partners (.bank, .com domains)
- AUTHORIZED: Invoice/financial reporting to approved vendors
- VIOLATION: Financial data to personal emails (gmail, outlook, etc.)
- VIOLATION: Banking credentials to unauthorized recipients
- OBFUSCATED: Encoded account numbers, hidden financial data

Generate JSON with this EXACT structure:
{{
  "channel": "email",
  "user": {{
    "role": "FINANCE", 
    "dept": "ACCOUNTING", 
    "seniority": "MANAGER"
  }},
  "recipients": ["billing@commercialbank.com", "payments@vendorcorp.net"],
  "subject": "Q3 Payment Processing - Account #4567890123", 
  "body": "Comprehensive financial communication with naturally embedded sensitive information like account numbers, routing numbers, payment amounts, credit card details, SSNs for tax purposes, banking credentials, financial projections, etc. Make it authentic with proper financial language, account references, and business context.",
  "attachments": [
    {{
      "name": "payment_batch_Q3.xlsx",
      "size": 892160,
      "mime_type": "application/vnd.ms-excel",
      "content_summary": "Batch payment spreadsheet containing vendor account numbers and routing details",
      "sensitivity_indicators": ["contains_banking_data", "payment_credentials", "account_numbers"]
    }},
    {{
      "name": "financial_summary_sept.pdf",
      "size": 156800,
      "mime_type": "application/pdf", 
      "content_summary": "Monthly financial report with revenue projections and account summaries",
      "sensitivity_indicators": ["financial_data", "revenue_projections"]
    }}
  ],
  "links": ["https://banking.commercialbank.com/corporate/payments"],
  "context_summary": "Quarterly payment processing coordination with banking partner. Discussion of payment credentials, account details, and financial data transfers.",
  "thread": {{
    "id_hash": "fin_thread_001",
    "age_days": 1,
    "prior_msgs": 0
  }}
}}

CRITICAL: Embed realistic sensitive information naturally in the body text including:
- Bank account numbers (e.g., "Account #4567890123")
- Routing numbers and SWIFT codes
- Credit card or PAN numbers (valid format)
- Payment amounts and financial figures
- Tax ID numbers and SSNs
- Banking login credentials
- Financial projections and sensitive data"""

        system_prompt = "You are an expert finance professional generating realistic financial communications."

        response = await self._generate_with_llm(prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_finance_example(data):
                self.generation_stats["successful"] += 1
                return GeneratedExample(
                    channel=data.get("channel", "email"),
                    user=data.get("user", {}),
                    recipients=data.get("recipients", []),
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    attachments=data.get("attachments", []),
                    links=data.get("links", []),
                    thread=data.get("thread"),
                    labels=data.get("labels"),
                    spans=data.get("spans"),
                    meta={"agent": "finance", **data.get("meta", {})}
                )

        self.generation_stats["failed"] += 1
        return None

    def _validate_finance_example(self, data: Dict) -> bool:
        """Validate finance-specific requirements."""
        body = data.get("body", "")
        finance_terms = ["payment", "invoice", "credit", "bank", "account", "wire"]
        has_finance_context = any(term in body.lower() for term in finance_terms)

        user = data.get("user", {})
        is_finance_role = user.get("role") in ["FINANCE", "ACCOUNTING", "TREASURY", "PAYMENTS"]

        # Validate attachment structure
        attachments = data.get("attachments", [])
        if not self._validate_attachment_schema(attachments):
            return False
            
        # Validate finance-specific sensitivity indicators
        finance_indicators = ["payment_data", "banking_info", "financial_records", "credit_card_data",
                            "wire_instructions", "account_numbers", "financial_pii", "treasury_data",
                            "invoice_details", "tax_information"]
        if not self._validate_sensitivity_indicators(attachments, finance_indicators):
            return False

        return has_finance_context and is_finance_role

    def _validate_attachment_schema(self, attachments: list) -> bool:
        """Validate that all attachments have proper schema structure."""
        if not attachments:
            return True  # Empty attachments list is valid
            
        for attachment in attachments:
            if not isinstance(attachment, dict):
                return False
                
            required_fields = ["name", "size", "mime_type", "content_summary", "sensitivity_indicators"]
            for field in required_fields:
                if field not in attachment:
                    return False
                    
            # Validate field types
            if not isinstance(attachment["name"], str) or not attachment["name"]:
                return False
            if not isinstance(attachment["size"], int) or attachment["size"] <= 0:
                return False
            if not isinstance(attachment["mime_type"], str) or not attachment["mime_type"]:
                return False
            if not isinstance(attachment["content_summary"], str) or not attachment["content_summary"]:
                return False
            if not isinstance(attachment["sensitivity_indicators"], list) or not attachment["sensitivity_indicators"]:
                return False
                
        return True

    def _validate_sensitivity_indicators(self, attachments: list, valid_indicators: list) -> bool:
        """Validate that attachments use appropriate sensitivity indicators for the domain."""
        for attachment in attachments:
            if isinstance(attachment, dict):
                indicators = attachment.get("sensitivity_indicators", [])
                if not any(indicator in valid_indicators for indicator in indicators):
                    return False
        return True


class HRAgent(BaseLLMAgent):
    """HR domain specialist - Employee communications, personal information."""

    def __init__(self, config):
        super().__init__(config, "HR")

    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        prompt = f"""Generate realistic HR communication for HRM-DLP training.

SCENARIO: {request.scenario_context}
RISK LEVEL: {request.risk_level}
TARGET SPANS: {request.target_spans}

Create a detailed HR communication with the following requirements:
1. Realistic business context (employee management, benefits, hiring, performance)
2. Natural inclusion of employee sensitive information
3. Appropriate HR terminology and processes
4. CRITICAL: Generate complete attachment metadata including realistic file sizes, detailed content summaries, and contextual sensitivity indicators
5. Context summary for multi-task learning

ATTACHMENT GENERATION REQUIREMENTS - CRITICAL:
- Include 1-3 attachments that fit the HR scenario
- Generate realistic file sizes (PDFs: 100KB-1MB, Excel: 50KB-500KB, Word docs: 50KB-300KB)
- Write detailed content summaries that describe what employee data the file contains
- Add 2-4 sensitivity indicators per attachment from: [contains_ssn, salary_information, personal_data, background_check, medical_information, family_details, performance_review, employee_records, benefits_data]
- **MANDATORY**: Each attachment MUST be a complete object with name, size, mime_type, content_summary, and sensitivity_indicators
- **FAILURE TO PROVIDE RICH ATTACHMENT METADATA WILL RESULT IN REJECTION**

RISK PATTERNS:
- AUTHORIZED: Employee data sharing with approved HR systems/vendors
- AUTHORIZED: Benefits enrollment with trusted insurance partners
- VIOLATION: Employee SSNs/personal data to personal emails (gmail, outlook, etc.)
- VIOLATION: Background check results to unauthorized recipients
- OBFUSCATED: Hidden employee data, encoded personal information

Generate JSON with this EXACT structure:
{{
  "channel": "email",
  "user": {{
    "role": "HR", 
    "dept": "PEOPLE", 
    "seniority": "SPECIALIST"
  }},
  "recipients": ["manager@company.com", "benefits@hrpartner.com"],
  "subject": "Employee Onboarding - John Smith (SSN: ***-**-1234)",
  "body": "Comprehensive HR communication with naturally embedded sensitive information like SSNs, employee IDs, salary information, background check results, medical information, emergency contacts, home addresses, phone numbers, etc. Make it authentic with proper HR language, employee references, and business context.",
  "attachments": [
    {{
      "name": "employee_record_jsmith.pdf",
      "size": 312320,
      "mime_type": "application/pdf",
      "content_summary": "Complete employee record containing SSN, salary, background check, emergency contacts",
      "sensitivity_indicators": ["contains_ssn", "salary_information", "personal_data", "background_check"]
    }},
    {{
      "name": "benefits_enrollment_2024.xlsx",
      "size": 205824,
      "mime_type": "application/vnd.ms-excel",
      "content_summary": "Benefits enrollment form with employee personal details and medical information",
      "sensitivity_indicators": ["medical_information", "personal_data", "family_details"]
    }}
  ],
  "links": ["https://hr.company.com/employee/profile/jsmith"],
  "context_summary": "New employee onboarding process involving transfer of sensitive employee data including SSN, salary, and personal information for benefits enrollment.",
  "thread": {{
    "id_hash": "hr_thread_001",
    "age_days": 0,
    "prior_msgs": 1
  }}
}}

CRITICAL: Embed realistic sensitive information naturally in the body text including:
- Social Security Numbers (e.g., "SSN: 123-45-6789")
- Employee ID numbers
- Salary and compensation details
- Home addresses and phone numbers
- Emergency contact information
- Medical/health information
- Background check results
- Performance review details"""

        system_prompt = "You are an expert HR professional generating realistic employee communications."

        response = await self._generate_with_llm(prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_hr_example(data):
                self.generation_stats["successful"] += 1
                return GeneratedExample(
                    channel=data.get("channel", "email"),
                    user=data.get("user", {}),
                    recipients=data.get("recipients", []),
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    attachments=data.get("attachments", []),
                    links=data.get("links", []),
                    thread=data.get("thread"),
                    labels=data.get("labels"),
                    spans=data.get("spans"),
                    meta={"agent": "hr", **data.get("meta", {})}
                )

        self.generation_stats["failed"] += 1
        return None

    def _validate_hr_example(self, data: Dict) -> bool:
        """Validate HR-specific requirements."""
        body = data.get("body", "")
        hr_terms = ["employee", "onboarding", "benefits", "performance", "salary"]
        has_hr_context = any(term in body.lower() for term in hr_terms)

        user = data.get("user", {})
        is_hr_role = user.get("role") in ["HR", "PEOPLE", "RECRUITING", "BENEFITS"]

        # Validate attachment structure
        attachments = data.get("attachments", [])
        if not self._validate_attachment_schema(attachments):
            return False
            
        # Validate HR-specific sensitivity indicators
        hr_indicators = ["personal_data", "employee_pii", "salary_information", "benefits_data",
                        "performance_records", "disciplinary_actions", "medical_information",
                        "background_check", "employment_history", "hr_records"]
        if not self._validate_sensitivity_indicators(attachments, hr_indicators):
            return False

        return has_hr_context and is_hr_role

    def _validate_attachment_schema(self, attachments: list) -> bool:
        """Validate that all attachments have proper schema structure."""
        if not attachments:
            return True  # Empty attachments list is valid
            
        for attachment in attachments:
            if not isinstance(attachment, dict):
                return False
                
            required_fields = ["name", "size", "mime_type", "content_summary", "sensitivity_indicators"]
            for field in required_fields:
                if field not in attachment:
                    return False
                    
            # Validate field types
            if not isinstance(attachment["name"], str) or not attachment["name"]:
                return False
            if not isinstance(attachment["size"], int) or attachment["size"] <= 0:
                return False
            if not isinstance(attachment["mime_type"], str) or not attachment["mime_type"]:
                return False
            if not isinstance(attachment["content_summary"], str) or not attachment["content_summary"]:
                return False
            if not isinstance(attachment["sensitivity_indicators"], list) or not attachment["sensitivity_indicators"]:
                return False
                
        return True

    def _validate_sensitivity_indicators(self, attachments: list, valid_indicators: list) -> bool:
        """Validate that attachments use appropriate sensitivity indicators for the domain."""
        for attachment in attachments:
            if isinstance(attachment, dict):
                indicators = attachment.get("sensitivity_indicators", [])
                if not any(indicator in valid_indicators for indicator in indicators):
                    return False
        return True


class SecurityAgent(BaseLLMAgent):
    """Security domain specialist - API keys, secrets, security incidents."""

    def __init__(self, config):
        super().__init__(config, "Security")

    async def generate_example(self, request: GenerationRequest) -> Optional[GeneratedExample]:
        prompt = f"""Generate realistic security communication for HRM-DLP training.

SCENARIO: {request.scenario_context}
RISK LEVEL: {request.risk_level}
TARGET SPANS: {request.target_spans}

Create a detailed security communication with the following requirements:
1. Realistic business context (API keys, credentials, security incidents)
2. Natural inclusion of security sensitive information
3. Appropriate security terminology and processes
4. CRITICAL: Generate complete attachment metadata including realistic file sizes, detailed content summaries, and contextual sensitivity indicators
5. Context summary for multi-task learning

ATTACHMENT GENERATION REQUIREMENTS - CRITICAL:
- Include 1-3 attachments that fit the security scenario
- Generate realistic file sizes (Text files: 2KB-50KB, PDFs: 100KB-1MB, Logs: 10KB-2MB)
- Write detailed content summaries that describe what security data the file contains
- Add 2-4 sensitivity indicators per attachment from: [contains_api_keys, production_credentials, database_access, security_incident, contains_credentials, internal_investigation, system_logs, access_tokens, private_keys]
- **MANDATORY**: Each attachment MUST be a complete object with name, size, mime_type, content_summary, and sensitivity_indicators
- **FAILURE TO PROVIDE RICH ATTACHMENT METADATA WILL RESULT IN REJECTION**

RISK PATTERNS:
- AUTHORIZED: API key sharing within approved DevOps/Security teams
- AUTHORIZED: Credential rotation with trusted infrastructure partners
- VIOLATION: API keys/secrets to personal emails (gmail, outlook, etc.)
- VIOLATION: Database credentials to unauthorized recipients
- OBFUSCATED: Base64-encoded secrets, hidden credentials

Generate JSON with this EXACT structure:
{{
  "channel": "slack",
  "user": {{
    "role": "SECURITY", 
    "dept": "INFOSEC", 
    "seniority": "ENGINEER"
  }},
  "recipients": ["@devops-team", "sre@company.com"],
  "subject": "🔐 API Key Rotation - Production Database Access",
  "body": "Comprehensive security communication with naturally embedded sensitive information like API keys, database URIs, access tokens, SSH keys, certificates, passwords, service account credentials, etc. Make it authentic with proper security language, incident references, and business context.",
  "attachments": [
    {{
      "name": "api_keys_prod.txt",
      "size": 4096,
      "mime_type": "text/plain",
      "content_summary": "Production API keys and access tokens for database and service authentication",
      "sensitivity_indicators": ["contains_api_keys", "production_credentials", "database_access"]
    }},
    {{
      "name": "incident_report_2024-089.pdf",
      "size": 184320,
      "mime_type": "application/pdf",
      "content_summary": "Security incident report detailing credential compromise and remediation steps",
      "sensitivity_indicators": ["security_incident", "contains_credentials", "internal_investigation"]
    }}
  ],
  "links": ["https://vault.company.com/secrets/production"],
  "context_summary": "Security team coordinating API key rotation for production systems. Discussion involves sharing sensitive credentials and access tokens for infrastructure management.",
  "thread": {{
    "id_hash": "sec_thread_001",
    "age_days": 2,
    "prior_msgs": 3
  }}
}}

CRITICAL: Embed realistic sensitive information naturally in the body text including:
- API keys (e.g., "AKIA1234567890ABCDEF")
- Database connection strings (e.g., "postgresql://user:pass@db.com:5432/prod")
- Access tokens and secrets
- SSH private key snippets
- Password credentials
- Service account information
- Certificate details"""

        system_prompt = "You are an expert cybersecurity professional generating realistic security communications."

        response = await self._generate_with_llm(prompt, system_prompt)
        if response:
            data = self._extract_json_from_response(response)
            if data and self._validate_security_example(data):
                self.generation_stats["successful"] += 1
                return GeneratedExample(
                    channel=data.get("channel", "email"),
                    user=data.get("user", {}),
                    recipients=data.get("recipients", []),
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    attachments=data.get("attachments", []),
                    links=data.get("links", []),
                    thread=data.get("thread"),
                    labels=data.get("labels"),
                    spans=data.get("spans"),
                    meta={"agent": "security", **data.get("meta", {})}
                )

        self.generation_stats["failed"] += 1
        return None

    def _validate_security_example(self, data: Dict) -> bool:
        """Validate security-specific requirements."""
        body = data.get("body", "")
        security_terms = ["credential", "api", "key", "secret", "token", "access"]
        has_security_context = any(term in body.lower() for term in security_terms)

        user = data.get("user", {})
        is_security_role = user.get("role") in ["SECURITY", "DEVOPS", "INFOSEC", "SRE"]

        # Validate attachment structure
        attachments = data.get("attachments", [])
        if not self._validate_attachment_schema(attachments):
            return False
            
        # Validate security-specific sensitivity indicators
        security_indicators = ["api_keys", "credentials", "secret_keys", "access_tokens",
                             "database_credentials", "security_logs", "incident_data",
                             "vulnerability_info", "system_access", "config_secrets"]
        if not self._validate_sensitivity_indicators(attachments, security_indicators):
            return False

        return has_security_context and is_security_role

    def _validate_attachment_schema(self, attachments: list) -> bool:
        """Validate that all attachments have proper schema structure."""
        if not attachments:
            return True  # Empty attachments list is valid
            
        for attachment in attachments:
            if not isinstance(attachment, dict):
                return False
                
            required_fields = ["name", "size", "mime_type", "content_summary", "sensitivity_indicators"]
            for field in required_fields:
                if field not in attachment:
                    return False
                    
            # Validate field types
            if not isinstance(attachment["name"], str) or not attachment["name"]:
                return False
            if not isinstance(attachment["size"], int) or attachment["size"] <= 0:
                return False
            if not isinstance(attachment["mime_type"], str) or not attachment["mime_type"]:
                return False
            if not isinstance(attachment["content_summary"], str) or not attachment["content_summary"]:
                return False
            if not isinstance(attachment["sensitivity_indicators"], list) or not attachment["sensitivity_indicators"]:
                return False
                
        return True

    def _validate_sensitivity_indicators(self, attachments: list, valid_indicators: list) -> bool:
        """Validate that attachments use appropriate sensitivity indicators for the domain."""
        for attachment in attachments:
            if isinstance(attachment, dict):
                indicators = attachment.get("sensitivity_indicators", [])
                if not any(indicator in valid_indicators for indicator in indicators):
                    return False
        return True
