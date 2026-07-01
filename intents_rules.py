import re

# Extended keyword rules to cover more intents and rare cases
RULES = [
    ("theatre_info",        [r"\btheatre\b", r"\bmovie\b", r"\bimax\b"]),
    ("store_info",          [r"\bstore\b", r"\bshop\b", r"\bbrand\b", r"\blocation\b"]),
    ("books_store_info",    [r"\bbook\b", r"\blibrary\b"]),
    ("home_store_info",     [r"\bhome\b", r"\bfurniture\b", r"\bkitchen\b"]),
    ("restaurant_info",     [r"\brestaurant\b", r"\bdine\b", r"\bfood\b"]),
    ("restaurant_location", [r"\brestaurant\b", r"\blocation\b", r"\bfloor\b"]),
    ("store_category",      [r"\bcategory\b", r"\btype\b", r"\bdepartment\b"]),
    ("store_discount",      [r"\boffer\b", r"\bdiscount\b", r"\bsale\b"]),
    ("parking_info",        [r"\bparking\b", r"\bgarage\b"]),
    ("atm_info",            [r"\batm\b", r"\bcash\b", r"\bbank\b"]),
    ("restroom_info",       [r"\brestroom\b", r"\btoilet\b", r"\bwashroom\b"]),
    ("lift_info",           [r"\blift\b", r"\belevator\b"]),
    ("escalator_info",      [r"\bescalator\b"]),
    ("exit_info",           [r"\bexit\b", r"\bway out\b"]),
    ("entry_info",          [r"\bentry\b", r"\bentrance\b"]),
    ("foodcourt_info",      [r"\bfood court\b", r"\bcafeteria\b", r"\bdining\b"]),
    ("general_faq",         [r".*"])  # fallback catch-all
]


def get_intent(text: str) -> str:
    """Return intent based on keyword rules"""
    if not text:
        return "general_faq"
    t = text.lower()
    for label, patterns in RULES:
        for pat in patterns:
            if re.search(pat, t):
                return label
    return "general_faq"

