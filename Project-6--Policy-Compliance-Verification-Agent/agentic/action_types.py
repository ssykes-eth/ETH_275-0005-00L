"""The catalogue of dashboard actions (provided scaffolding).

Each action type declares: a human label, the **policy document** that governs it
(used to focus retrieval), and a field schema (used to render the dashboard form,
validate input, and summarise the action for the agents).

The field schemas line up with real thresholds in ``rag/data`` — e.g. the expense
hotel/receipt/deadline rules, the procurement approval thresholds, the InfoSec MFA
requirement, the leave notice periods — so the verifier has genuine rules to catch.
"""

from __future__ import annotations

ACTION_TYPES: dict[str, dict] = {
    "expense_report": {
        "label": "Expense Report",
        "policy_source": "expense_reimbursement_policy.md",
        "description": "Submit a business expense for reimbursement.",
        "fields": [
            {"name": "category", "label": "Category", "type": "select", "required": True,
             "options": ["hotel", "meal", "flight", "taxi", "client_entertainment", "other"]},
            {"name": "amount", "label": "Amount (CHF)", "type": "number", "required": True},
            {"name": "region", "label": "Region", "type": "select", "required": True,
             "options": ["switzerland", "western_europe", "north_america", "asia_pacific", "other"]},
            {"name": "receipt_attached", "label": "Receipt attached", "type": "boolean", "required": True},
            {"name": "days_since_expense", "label": "Days since expense", "type": "number", "required": True},
            {"name": "description", "label": "Description", "type": "text", "required": False},
        ],
    },
    "procurement_request": {
        "label": "Procurement Request",
        "policy_source": "procurement_policy.md",
        "description": "Request approval to purchase goods or services from a vendor.",
        "fields": [
            {"name": "amount", "label": "Total contract value (CHF)", "type": "number", "required": True},
            {"name": "num_quotes", "label": "Number of quotes obtained", "type": "number", "required": True},
            {"name": "approver", "label": "Approver", "type": "select", "required": True,
             "options": ["direct_manager", "department_head", "finance_director", "ceo"]},
            {"name": "purchase_order_raised", "label": "Purchase order raised", "type": "boolean", "required": True},
            {"name": "vendor_name", "label": "Vendor", "type": "text", "required": True},
            {"name": "description", "label": "What is being purchased", "type": "text", "required": False},
        ],
    },
    "access_change": {
        "label": "Access / Config Change",
        "policy_source": "information_security_policy.md",
        "description": "Grant access to a system or change a security configuration.",
        "fields": [
            {"name": "resource_classification", "label": "Data classification", "type": "select", "required": True,
             "options": ["public", "internal", "confidential", "restricted"]},
            {"name": "mfa_enabled", "label": "MFA enabled", "type": "boolean", "required": True},
            {"name": "shared_account", "label": "Shared account", "type": "boolean", "required": True},
            {"name": "grant_to", "label": "Grant access to", "type": "text", "required": True},
            {"name": "justification", "label": "Business justification", "type": "text", "required": False},
        ],
    },
    "time_off_request": {
        "label": "Time-off Request",
        "policy_source": "leave_and_absence_policy.md",
        "description": "Request leave / time off.",
        "fields": [
            {"name": "leave_type", "label": "Leave type", "type": "select", "required": True,
             "options": ["annual", "sick", "maternity", "paternity", "study"]},
            {"name": "days", "label": "Number of days", "type": "number", "required": True},
            {"name": "notice_days", "label": "Notice given (days ahead)", "type": "number", "required": True},
            {"name": "medical_certificate", "label": "Medical certificate attached", "type": "boolean", "required": False},
            {"name": "start_date", "label": "Start date", "type": "text", "required": False},
        ],
    },
}


def get_action_type(action_type: str) -> dict | None:
    """The schema for ``action_type``, or ``None`` if it isn't a known type."""
    return ACTION_TYPES.get(action_type)


def required_fields(action_type: str) -> list[str]:
    """Names of the required fields for an action type ([] if the type is unknown)."""
    spec = ACTION_TYPES.get(action_type)
    if not spec:
        return []
    return [f["name"] for f in spec["fields"] if f.get("required")]


def policy_source_for(action_type: str) -> str | None:
    """The governing policy filename for an action type, if known."""
    spec = ACTION_TYPES.get(action_type)
    return spec["policy_source"] if spec else None
