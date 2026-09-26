"""Label sets and descriptions: the single source of truth both systems are given.

Jev receives these as Choice/Score criteria; the LLMs receive the same text in the
prompt plus a JSON-schema enum. Descriptions were written from the dev split only.
Changing any text here changes the request hash, so it creates new cache entries.
"""

# --- Banking77: one Choice over 77 intents ---------------------------------------

BANKING77_INTENTS: dict[str, str] = {
    "Refund_not_showing_up": "A refund the customer expected has not appeared in their account or statement",
    "activate_my_card": "How to activate a new card, or problems activating it",
    "age_limit": "Minimum age or age requirements to open or use an account",
    "apple_pay_or_google_pay": "Using or topping up with Apple Pay or Google Pay",
    "atm_support": "General questions about withdrawing cash at ATMs and where ATMs can be used",
    "automatic_top_up": "Setting up or understanding automatic (auto) top-up",
    "balance_not_updated_after_bank_transfer": "Account balance has not updated after a bank transfer into the account",
    "balance_not_updated_after_cheque_or_cash_deposit": "Balance has not updated after depositing a cheque or cash",
    "beneficiary_not_allowed": "A transfer failed because the recipient or beneficiary is not allowed or cannot be added",
    "cancel_transfer": "Cancelling or reversing a transfer the customer made, e.g. by mistake",
    "card_about_to_expire": "Card is expiring soon; ordering a renewal or replacement",
    "card_acceptance": "Where or in which shops and countries the card is accepted",
    "card_arrival": "An ordered card has not arrived yet; tracking a card in the mail",
    "card_delivery_estimate": "How long card delivery takes, or choosing delivery time or location",
    "card_linking": "Linking or adding an existing card to the app or account",
    "card_not_working": "A physical card does not work at all, cause unknown",
    "card_payment_fee_charged": "An unexpected fee was charged for paying by card",
    "card_payment_not_recognised": "A card payment appears that the customer did not make or does not recognise",
    "card_payment_wrong_exchange_rate": "The wrong exchange rate was applied to a card purchase",
    "card_swallowed": "The ATM kept or did not return the card",
    "cash_withdrawal_charge": "A fee was charged for withdrawing cash",
    "cash_withdrawal_not_recognised": "A cash withdrawal appears that the customer did not make",
    "change_pin": "How to change or set a new PIN",
    "compromised_card": "The card may be compromised or used by someone else, without being lost",
    "contactless_not_working": "Contactless payment does not work or how to use contactless",
    "country_support": "Which countries are supported or where the customer must live to use the service",
    "declined_card_payment": "A card payment in a shop or online was declined or rejected",
    "declined_cash_withdrawal": "An ATM declined the cash withdrawal",
    "declined_transfer": "A transfer was declined",
    "direct_debit_payment_not_recognised": "A direct debit appears that the customer does not recognise",
    "disposable_card_limits": "Limits on disposable virtual cards, such as number of uses or cards",
    "edit_personal_details": "Changing name, address or other personal details",
    "exchange_charge": "Fees or costs for exchanging currency",
    "exchange_rate": "What the current exchange rates are or how they are set",
    "exchange_via_app": "Exchanging currencies inside the app and which currencies can be exchanged",
    "extra_charge_on_statement": "An unexplained small extra charge on the statement",
    "failed_transfer": "A transfer failed or cannot be made",
    "fiat_currency_support": "Which fiat currencies the account can hold or use, e.g. when travelling",
    "get_disposable_virtual_card": "How to get or how disposable virtual cards work",
    "get_physical_card": "Getting the physical card set up, including its PIN, after ordering",
    "getting_spare_card": "Ordering an additional or spare card, e.g. for a family member",
    "getting_virtual_card": "How to get or where to find a (non-disposable) virtual card",
    "lost_or_stolen_card": "The card was lost or stolen",
    "lost_or_stolen_phone": "The phone with the app was lost or stolen",
    "order_physical_card": "How to order a physical card",
    "passcode_forgotten": "Forgot or needs to reset the app passcode",
    "pending_card_payment": "A card payment is still showing as pending",
    "pending_cash_withdrawal": "A cash withdrawal is still showing as pending",
    "pending_top_up": "A top-up is still pending or slow to process",
    "pending_transfer": "A transfer is still pending and not yet available",
    "pin_blocked": "The PIN is blocked after too many wrong attempts",
    "receiving_money": "How to receive money from others, including salary or other currencies",
    "request_refund": "The customer asks for a refund or reversal of a payment",
    "reverted_card_payment?": "A card payment was reverted or returned",
    "supported_cards_and_currencies": "Which cards and currencies can be used to add money",
    "terminate_account": "Closing or deleting the account",
    "top_up_by_bank_transfer_charge": "Fees for topping up by bank transfer (e.g. SEPA, SWIFT)",
    "top_up_by_card_charge": "Fees for topping up with a card",
    "top_up_by_cash_or_cheque": "Topping up with cash or a cheque",
    "top_up_failed": "A top-up failed",
    "top_up_limits": "Limits on how much or how often the customer can top up",
    "top_up_reverted": "A top-up was reverted or cancelled",
    "topping_up_by_card": "Topping up with a card, including a card top-up not showing yet",
    "transaction_charged_twice": "The same transaction was charged twice or more",
    "transfer_fee_charged": "A fee was charged for making a transfer",
    "transfer_into_account": "How to transfer money into the account from a bank account",
    "transfer_not_received_by_recipient": "A completed transfer has not been received by the recipient",
    "transfer_timing": "How long a transfer takes to arrive",
    "unable_to_verify_identity": "The customer tried but cannot complete identity verification",
    "verify_my_identity": "How to verify identity and what documents are needed",
    "verify_source_of_funds": "Verifying or being asked about the source of funds",
    "verify_top_up": "Why or how a top-up needs to be verified, e.g. a verification code",
    "virtual_card_not_working": "A virtual card does not work",
    "visa_or_mastercard": "Whether cards are Visa or Mastercard, or getting a specific one",
    "why_verify_identity": "Why identity verification is required",
    "wrong_amount_of_cash_received": "An ATM dispensed a different amount than was charged",
    "wrong_exchange_rate_for_cash_withdrawal": "The wrong exchange rate or amount for a cash withdrawal in a foreign currency",
}

BANKING77_INSTRUCTION = "Which intent best describes this banking customer's message?"

# --- Tickets: queue / priority / type + demo signals -----------------------------

# Queue descriptions follow the dataset card's own definitions.
QUEUES: dict[str, str] = {
    "Technical Support": "Technical issues and support requests",
    "Customer Service": "Customer inquiries and service requests",
    "Billing and Payments": "Billing issues and payment processing",
    "Product Support": "Support for product-related issues",
    "IT Support": "Internal IT support and infrastructure issues",
    "Returns and Exchanges": "Product returns and exchanges",
    "Sales and Pre-Sales": "Sales inquiries and pre-sales questions",
    "Human Resources": "Employee inquiries and HR-related issues",
    "Service Outages and Maintenance": "Service interruptions and maintenance",
    "General Inquiry": "General inquiries and information requests",
}
QUEUE_INSTRUCTION = "Which support team should handle this ticket?"

# Ordered low -> high; Jev scores it, the LLMs pick one.
PRIORITIES: dict[str, str] = {
    "low": "Low: no immediate business impact; informational, or can wait",
    "medium": "Medium: noticeable impact but work can continue or a workaround exists",
    "high": "High: significant impact; critical systems, security, data or money at risk; needs prompt action",
}
PRIORITY_INSTRUCTION = "How urgent is this ticket?"

# ITIL ticket types. A dataset-convention rewrite was tried and reverted
# (tasks/eval_ledger.md, iteration 1).
TYPES: dict[str, str] = {
    "Incident": "Something that was working is broken or disrupted right now",
    "Problem": "An underlying or recurring issue whose root cause needs investigation",
    "Request": "A request for information, help, access or a standard service",
    "Change": "A request to modify, add or improve a feature, system or configuration",
}
TYPE_INSTRUCTION = "What type of ticket is this?"

# Yes/no signals shown in the demo (no gold labels; not scored).
SIGNALS: dict[str, str] = {
    "security_or_data_breach": "The ticket reports a security incident, breach, or exposure of data",
    "service_outage": "The ticket reports that a service or system is down or unavailable",
    "customer_frustrated": "The customer sounds frustrated or upset",
    "asks_for_refund": "The customer asks for a refund or money back",
}
