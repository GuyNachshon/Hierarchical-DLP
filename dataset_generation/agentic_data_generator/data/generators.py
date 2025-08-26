"""
Email and attachment content generators for realistic DLP training data.
"""

import random
from typing import Tuple, List, Optional, Dict


class EmailAddressGenerator:
    """Generates diverse, realistic email addresses for DLP training data."""

    def __init__(self):
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Dorothy", "Mark", "Sandra", "Donald", "Donna",
            "Steven", "Carol", "Paul", "Ruth", "Andrew", "Sharon", "Joshua", "Michelle",
            "Kenneth", "Laura", "Kevin", "Emily", "Brian", "Kimberly", "George", "Deborah",
            "Timothy", "Amy", "Ronald", "Angela", "Jason", "Ashley", "Edward", "Brenda",
            "Jeffrey", "Emma", "Ryan", "Olivia", "Jacob", "Cynthia", "Gary", "Marie",
            "Ahmed", "Priya", "Carlos", "Li", "Kumar", "Sofia", "Hassan", "Elena",
            "Raj", "Anna", "Diego", "Maria", "Alex", "Chen", "Nina", "Jose",
            "Amit", "Rosa", "Omar", "Fatima", "Ravi", "Lucia", "Ibrahim", "Grace"
        ]

        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
            "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
            "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
            "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
            "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
            "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
            "Patel", "Wang", "Kim", "Singh", "Chen", "Liu", "Zhang", "Ali",
            "Kumar", "Shah", "Gupta", "Mehta", "Sharma", "Zhao", "Yang", "Wu",
            "Ahmed", "Hassan", "Rahman", "Khan", "Luo", "Cheng", "Zhou", "Xu"
        ]

        self.domain_pools = {
            "legal": [
                "smithjohnsonlaw.com", "litigation-partners.net", "corporatelaw-llc.com",
                "legal-counsel.org", "whiteshoe-law.com", "bakerlaw.net", "lawfirm-associates.com",
                "attorneygroup.com", "legalpartners.net", "counselors-law.com", "jurisprudence.org",
                "advocacy-law.com", "defensecounsel.net", "litigation-group.com", "legaladvice.org"
            ],
            "financial": [
                "commercialbank.com", "payments.wellsfargo.com", "processing.chase.com",
                "businessbank.net", "corporatefinance.org", "paymentgateway.com", "bankingpartner.net",
                "financialservices.com", "treasurymanagement.net", "creditprocessing.com", "fintechpartner.org",
                "paymentprocessor.net", "merchantservices.com", "bankcorp.net", "financialgroup.com"
            ],
            "hr_vendors": [
                "hrpartners.com", "benefitsplus.net", "payrollservices.org", "talentmanagement.com",
                "hrtech.net", "employeesolutions.com", "workforcepartners.net", "hrprovider.org",
                "benefits-corp.com", "hrmanagement.net", "talentacquisition.com", "payrollplus.org",
                "employeebenefits.net", "hrservices.com", "workplacesolutions.org"
            ],
            "vendors": [
                "acmesupplies.com", "officedepot.net", "businesssolutions.org", "vendorpartner.com",
                "suppliercorp.net", "businesspartners.com", "vendorservices.org", "supplychainplus.net",
                "corporatesupplies.com", "businessvendor.net", "vendorgroup.org", "suppliernetwork.com",
                "businesssource.net", "vendorcorp.com", "supplierpartners.org"
            ],
            "consulting": [
                "consultinggroup.com", "businessadvisors.net", "strategyconsulting.org", "managementpartners.com",
                "consultingfirm.net", "advisoryservices.com", "strategicpartners.org", "businessconsulting.net",
                "consultingcorp.com", "advisorygroup.net", "strategysolutions.org", "consultingservices.com"
            ],
            "tech_vendors": [
                "techsolutions.com", "cloudservices.net", "softwaredevelopment.org", "itpartners.com",
                "techcorp.net", "cloudcomputing.com", "softwarehouse.org", "itservices.net",
                "techpartners.com", "cloudsolutions.net", "softwarefirm.org", "itsolutions.com"
            ]
        }

    def generate_person_name(self) -> Tuple[str, str]:
        """Generate a random first and last name combination."""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        return first, last

    def generate_internal_email(self, dept: Optional[str] = None) -> str:
        """Generate realistic internal company email."""
        first, last = self.generate_person_name()
        patterns = [
            f"{first.lower()}.{last.lower()}@company.com",
            f"{first[0].lower()}{last.lower()}@company.com",
            f"{first.lower()}{last[0].lower()}@company.com",
        ]

        if dept and dept.lower() in ["finance", "accounting", "hr", "people", "legal"]:
            dept_lower = dept.lower()
            patterns.extend([
                f"{first.lower()}.{last.lower()}@{dept_lower}.company.com",
                f"{dept_lower}-team@company.com",
                f"{dept_lower}.group@company.com"
            ])

        return random.choice(patterns)

    def generate_external_email(self, business_type: str, role: Optional[str] = None) -> str:
        """Generate realistic external business email."""
        first, last = self.generate_person_name()

        if business_type not in self.domain_pools:
            business_type = "vendors"

        domain = random.choice(self.domain_pools[business_type])
        role_patterns = {
            "legal": ["counsel", "attorney", "legal", "partner"],
            "finance": ["billing", "payments", "finance", "accounting"],
            "hr": ["hr", "benefits", "recruiting", "talent"],
            "vendor": ["sales", "support", "billing", "contact"],
            "consulting": ["consulting", "advisor", "partner", "director"]
        }

        patterns = [
            f"{first.lower()}.{last.lower()}@{domain}",
            f"{first[0].lower()}{last.lower()}@{domain}",
            f"{first.lower()}{last[0].lower()}@{domain}"
        ]

        if role and role in role_patterns:
            role_prefixes = role_patterns[role]
            for prefix in role_prefixes:
                patterns.extend([
                    f"{prefix}@{domain}",
                    f"{first.lower()}.{prefix}@{domain}",
                    f"{prefix}.{last.lower()}@{domain}"
                ])

        return random.choice(patterns)

    def generate_personal_email(self) -> str:
        """Generate personal email addresses for leak scenarios."""
        first, last = self.generate_person_name()
        personal_domains = [
            "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
            "proton.me", "aol.com", "icloud.com"
        ]

        domain = random.choice(personal_domains)
        patterns = [
            f"{first.lower()}.{last.lower()}@{domain}",
            f"{first.lower()}{last.lower()}@{domain}",
            f"{first.lower()}.{last.lower()}{random.randint(1, 999)}@{domain}",
            f"{first[0].lower()}{last.lower()}@{domain}",
            f"{first.lower()}{random.randint(80, 99)}@{domain}"
        ]

        return random.choice(patterns)


class AttachmentContentGenerator:
    """Generates realistic attachment content and references for DLP training."""

    def __init__(self):
        self.attachment_types = {
            "legal": {
                "extensions": ["pdf", "docx", "doc"],
                "names": [
                    "nda_draft_{matter}.pdf", "contract_review_{client}.docx", "legal_opinion_{case}.pdf",
                    "settlement_agreement_{matter}.pdf", "due_diligence_{deal}.docx", "compliance_report_{year}.pdf",
                    "litigation_memo_{case}.docx", "regulatory_filing_{dept}.pdf", "patent_application_{tech}.pdf"
                ],
                "content_snippets": [
                    "As detailed in Section 3.2 of the attached NDA...",
                    "Please review the confidentiality clauses in the attached agreement (pages 4-7)...",
                    "The due diligence checklist in Exhibit A shows...",
                    "Per the settlement terms outlined in Attachment 1...",
                    "The legal analysis in the attached memorandum concludes..."
                ]
            },
            "finance": {
                "extensions": ["xlsx", "csv", "pdf"],
                "names": [
                    "payment_batch_{date}.xlsx", "invoice_details_{vendor}.pdf", "bank_reconciliation_{month}.xlsx",
                    "expense_report_{dept}_{month}.xlsx", "tax_documents_{year}.pdf", "audit_trail_{quarter}.csv",
                    "vendor_payments_{period}.xlsx", "budget_analysis_{dept}.xlsx", "financial_summary_{quarter}.pdf"
                ],
                "content_snippets": [
                    "See cell B12 in the attached spreadsheet for the payment amount...",
                    "The vendor details are listed in column C of the attached file...",
                    "Account numbers are shown in the 'Bank Info' tab (cells D5-D47)...",
                    "Reference the 'Payments' sheet for routing numbers...",
                    "Tax ID numbers are in the final column of the attached CSV..."
                ]
            },
            "hr": {
                "extensions": ["pdf", "docx", "xlsx"],
                "names": [
                    "employee_record_{name}.pdf", "background_check_{name}_{date}.pdf", "benefits_enrollment_{name}.docx",
                    "payroll_summary_{period}.xlsx", "performance_review_{name}_{year}.pdf", "disciplinary_action_{name}.pdf",
                    "onboarding_checklist_{name}.docx", "salary_analysis_{dept}.xlsx", "hr_metrics_{quarter}.pdf"
                ],
                "content_snippets": [
                    "The employee's SSN is shown in section 2 of the attached form...",
                    "Background check results are detailed in the attached report (pages 2-3)...",
                    "Salary information is in the 'Compensation' section of the attached document...",
                    "Medical records are referenced in Appendix B of the attached file...",
                    "Emergency contact details are listed at the bottom of page 1..."
                ]
            }
        }

    def generate_attachment_name(self, category: str, context: Dict[str, str] = None) -> str:
        """Generate realistic attachment filename."""
        if category not in self.attachment_types:
            category = "general"

        attachment_info = self.attachment_types[category]
        name_template = random.choice(attachment_info["names"])
        extension = random.choice(attachment_info["extensions"])

        context = context or {}
        replacements = {
            "{matter}": context.get("matter", f"M{random.randint(1000, 9999)}"),
            "{client}": context.get("client", random.choice(["acme", "globalcorp", "techsolutions"])),
            "{case}": context.get("case", f"CASE-{random.randint(100, 999)}"),
            "{deal}": context.get("deal", f"DEAL-{random.randint(100, 999)}"),
            "{year}": context.get("year", str(random.randint(2023, 2025))),
            "{date}": context.get("date", f"{random.randint(1, 12):02d}{random.randint(1, 28):02d}"),
            "{vendor}": context.get("vendor", random.choice(["supplier1", "vendor_corp", "services_inc"])),
            "{month}": context.get("month", f"{random.randint(1, 12):02d}"),
            "{dept}": context.get("dept", random.choice(["finance", "hr", "legal", "sales"])),
            "{quarter}": context.get("quarter", f"Q{random.randint(1, 4)}"),
            "{period}": context.get("period", f"{random.randint(1, 12):02d}-2024"),
            "{name}": context.get("name", f"{random.choice(['john_smith', 'jane_doe', 'alex_chen'])}"),
        }

        for placeholder, value in replacements.items():
            name_template = name_template.replace(placeholder, value)

        return f"{name_template.split('.')[0]}.{extension}"

    def generate_attachment_reference(self, category: str, attachment_name: str) -> str:
        """Generate realistic email content that references attachment."""
        if category not in self.attachment_types:
            category = "general"

        snippets = self.attachment_types[category]["content_snippets"]
        return random.choice(snippets)