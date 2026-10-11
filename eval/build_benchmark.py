"""Build the labeled synthetic benchmark in eval/benchmark/ and eval/labels.json.

Everything is synthetic and generated from a fixed seed, so running this again
reproduces the same files byte for byte (apart from .xlsx zip timestamps):

- SSNs are random numbers in the valid format (area not 000, 666 or 900-999,
  group not 00, serial not 0000). They are not taken from any real record.
- Card numbers are only the publicly documented test numbers that payment
  processors publish for testing (they are not issued to anyone).
- Routing numbers are random with a valid ABA check digit; account numbers are random.
- Names are invented; phone numbers use the fictional 555-01xx range.

Each label's `value` is an exact substring of the document text (for .xlsx, of
a cell). See eval/README.md for the matching rules used by scripts/evaluate.py.
"""
import json
import os
import random

from openpyxl import Workbook

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 20261011
N_DOCS = 48  # 24 dev + 24 held-out

FIRST = ["Avery", "Jordan", "Riley", "Casey", "Morgan", "Quinn", "Rowan", "Skyler", "Emerson", "Hayden", "Parker", "Reese"]
LAST = ["Lindqvist", "Okafor", "Marchetti", "Delacroix", "Nakamura", "Abernathy", "Castellano", "Whitford", "Vasquez", "Ostrowski"]
TEST_CARDS = [
    "4111 1111 1111 1111", "5555 5555 5555 4444", "3782 822463 10005", "6011 1111 1111 1117",
    "4012-8888-8888-1881", "5105105105105100", "3714 496353 98431", "4242 4242 4242 4242",
]


class Doc:
    def __init__(self, rng):
        self.rng = rng
        self.lines = []
        self.labels = []

    def add(self, line, labels=()):
        for ftype, category, value in labels:
            assert value in line, (value, line)
            self.labels.append({"type": ftype, "category": category, "value": value, "span": line})
        self.lines.append(line)


# ---------------------------------------------------------------- value generators

def name(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def valid_ssn(rng, sep="-"):
    area = rng.choice([a for a in range(1, 900) if a != 666])
    return f"{area:03d}{sep}{rng.randint(1, 99):02d}{sep}{rng.randint(1, 9999):04d}"


def invalid_ssn(rng):
    kind = rng.choice(["area000", "area666", "area9xx", "group00", "serial0000"])
    area = {"area000": 0, "area666": 666, "area9xx": rng.randint(900, 999)}.get(kind, rng.randint(1, 665))
    group = 0 if kind == "group00" else rng.randint(1, 99)
    serial = 0 if kind == "serial0000" else rng.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def aba_check_ok(d):
    n = [int(c) for c in d]
    return (3 * (n[0] + n[3] + n[6]) + 7 * (n[1] + n[4] + n[7]) + (n[2] + n[5] + n[8])) % 10 == 0


def with_aba_check(prefix8):
    for last in range(10):
        if aba_check_ok(prefix8 + str(last)):
            return prefix8 + str(last)
    raise AssertionError


def routing_number(rng):
    first2 = rng.choice(list(range(1, 13)) + list(range(21, 33)) + list(range(61, 73)))
    return with_aba_check(f"{first2:02d}{rng.randint(0, 999999):06d}")


def aba_valid_order_id(rng):
    """A 9-digit number that passes the ABA checksum but is an order ID (hard negative)."""
    return routing_number(rng)


def account_number(rng):
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(8, 12)))


def luhn_ok(digits):
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def luhn_tracking_number(rng):
    """16 digits starting with 9 (no card network) that pass Luhn (hard negative)."""
    while True:
        d = "9" + "".join(str(rng.randint(0, 9)) for _ in range(15))
        if luhn_ok(d):
            return d


def phone(rng):
    return rng.choice([f"(212) 555-01{rng.randint(0, 99):02d}", f"555-01{rng.randint(0, 99):02d}", f"+1 415 555 01{rng.randint(0, 99):02d}"])


def date(rng):
    m, d = rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([f"2026-{m:02d}-{d:02d}", f"{m:02d}/{d:02d}/2026"])


def money(rng, lo, hi, step=500):
    return f"${rng.randrange(lo, hi, step):,}"


# ---------------------------------------------------------------- sentence templates

def nppi_ssn(doc):
    r = doc.rng
    v = valid_ssn(r, sep=r.choice(["-", "-", " "]))
    doc.add(r.choice([
        f"Employee SSN: {v}",
        f"{name(r)} (social security number {v}) completed onboarding.",
        f"Please verify the SSN on file, {v}, before the W-2 run.",
    ]), [("NPPI", "ssn", v)])


def nppi_card(doc):
    r = doc.rng
    v = r.choice(TEST_CARDS)
    doc.add(r.choice([
        f"Corporate card number {v} expires 09/28.",
        f"Reimburse the charge on card {v} for the conference fee.",
        f"Payment card: {v}",
    ]), [("NPPI", "credit_card", v)])


def nppi_routing(doc):
    r = doc.rng
    v = routing_number(r)
    doc.add(r.choice([
        f"Direct deposit routing number: {v}",
        f"Bank ABA {v}, checking.",
        f"Use routing no. {v} for the payroll transfer.",
    ]), [("NPPI", "routing_number", v)])


def nppi_account(doc):
    r = doc.rng
    v = account_number(r)
    doc.add(r.choice([
        f"Checking account number: {v}",
        f"Deposit to acct #{v} at the credit union.",
        f"Savings account {v} was closed last month.",
    ]), [("NPPI", "bank_account", v)])


def hrci_compensation(doc):
    r = doc.rng
    amount = money(r, 62000, 185000)
    choice = r.randint(0, 2)
    if choice == 0:
        doc.add(f"{name(r)}'s base salary increases to {amount} effective {date(r)}.", [("HRCI", "compensation", amount)])
    elif choice == 1:
        bonus = money(r, 2000, 30000)
        doc.add(f"Approved annual bonus of {bonus} for {name(r)}.", [("HRCI", "compensation", bonus)])
    else:
        doc.add(f"Offer letter: starting salary {amount} plus 10% target bonus.", [("HRCI", "compensation", amount)])


def hrci_performance(doc):
    r = doc.rng
    v = r.choice(["performance improvement plan", "final written warning", "rated 'does not meet expectations'"])
    doc.add({
        "performance improvement plan": f"{name(r)} was placed on a performance improvement plan for 60 days.",
        "final written warning": f"{name(r)} received a final written warning for repeated absences.",
        "rated 'does not meet expectations'": f"In the annual review, {name(r)} was rated 'does not meet expectations'.",
    }[v], [("HRCI", "performance", v)])


def hrci_termination(doc):
    r = doc.rng
    v = r.choice(["employment will be terminated", "severance package", "involuntary termination"])
    doc.add({
        "employment will be terminated": f"{name(r)}'s employment will be terminated effective {date(r)}.",
        "severance package": f"HR approved a severance package of {money(r, 10000, 60000)} for {name(r)}.",
        "involuntary termination": f"Exit interview notes: involuntary termination after the investigation.",
    }[v], [("HRCI", "termination", v)])


def hrci_medical(doc):
    r = doc.rng
    v = r.choice(["medical leave", "workers' compensation claim", "disability accommodation"])
    doc.add({
        "medical leave": f"{name(r)} is on approved medical leave after knee surgery.",
        "workers' compensation claim": f"A workers' compensation claim was filed for a back injury on the loading dock.",
        "disability accommodation": f"{name(r)} requested a disability accommodation for a chronic condition.",
    }[v], [("HRCI", "medical", v)])


POSITIVES = [nppi_ssn, nppi_card, nppi_routing, nppi_account, hrci_compensation, hrci_performance, hrci_termination, hrci_medical]


def hard_negative(doc):
    r = doc.rng
    doc.add(r.choice([
        lambda: f"Call the help desk at {phone(r)} if the badge reader fails.",
        lambda: f"Order ID {aba_valid_order_id(r)} shipped on {date(r)}.",
        lambda: f"Invoice INV-{r.randint(100000, 999999)} is due {date(r)}.",
        lambda: f"Tracking number {luhn_tracking_number(r)} (ground shipping).",
        lambda: f"Form reference {invalid_ssn(r)} was archived.",
        lambda: f"SKU {r.randint(1000, 9999)}-{r.randint(10, 99)}-{r.randint(1000, 9999)} is out of stock.",
        lambda: f"Unit price {money(r, 5, 200, 1)}.99; quarterly sales revenue was $2.4M.",
        lambda: f"Server performance improved after the {date(r)} patch.",
        lambda: f"Please terminate the nightly cron job before {date(r)}.",
        lambda: f"Account manager: {name(r)}. Meeting at 10:30 in room 4{r.randint(10, 99)}.",
        lambda: f"Headcount is {r.randint(40, 900)} employees across {r.randint(2, 9)} offices.",
        lambda: f"Ticket #{r.randint(10000000, 99999999)} was resolved; ZIP 9{r.randint(1000, 9999)}-{r.randint(1000, 9999)}.",
    ])())


def build_text_doc(rng, n_pos):
    doc = Doc(rng)
    doc.add(rng.choice(["HR case notes (synthetic)", "Payroll update memo (synthetic)", "Operations email thread (synthetic)", "Finance desk notes (synthetic)"]))
    # Distinct categories per document, so every category shows up across many documents
    items = rng.sample(POSITIVES, n_pos) + [hard_negative] * rng.randint(2, 4)
    rng.shuffle(items)
    for item in items:
        item(doc)
    return doc


def build_xlsx_doc(rng, path, n_rows):
    """Payroll-like sheet plus a notes sheet. Headers give the context."""
    labels, wb = [], Workbook()
    ws = wb.active
    ws.title = rng.choice(["Payroll", "Employees", "Direct deposit"])
    ws.append(["Name", "Employee ID", "SSN", "Salary", "Routing", "Account", "Phone", "Start date"])
    for _ in range(n_rows):
        ssn = valid_ssn(rng)
        sal = money(rng, 55000, 190000)
        rout = routing_number(rng)
        acct = account_number(rng)
        row = [name(rng), f"E{rng.randint(10000, 99999)}", ssn, sal, rout, acct, phone(rng), date(rng)]
        ws.append(row)
        for ftype, cat, v in [("NPPI", "ssn", ssn), ("HRCI", "compensation", sal), ("NPPI", "routing_number", rout), ("NPPI", "bank_account", acct)]:
            labels.append({"type": ftype, "category": cat, "value": v, "span": v})
    notes = wb.create_sheet("Notes")
    notes.append(["Note"])
    note_doc = Doc(rng)
    for item in [rng.choice([hrci_performance, hrci_medical, hrci_termination]), hard_negative, hard_negative]:
        item(note_doc)
    for line in note_doc.lines:
        notes.append([line])
    labels.extend(note_doc.labels)
    wb.save(path)
    return labels


def main():
    rng = random.Random(SEED)
    out = {"seed": SEED, "documents": []}
    for i in range(N_DOCS):
        split = "dev" if i % 2 == 0 else "heldout"
        folder = os.path.join(ROOT, "benchmark", split)
        os.makedirs(folder, exist_ok=True)
        is_xlsx = i % 8 in (3, 6)  # 12 of 48
        name_ = f"doc{i + 1:02d}." + ("xlsx" if is_xlsx else "txt")
        path = os.path.join(folder, name_)
        if is_xlsx:
            labels = build_xlsx_doc(rng, path, n_rows=rng.randint(2, 4))
        elif i % 8 == 5:
            doc = Doc(rng)  # negatives only: nothing sensitive in the whole document
            doc.add("Facilities update (synthetic)")
            for _ in range(rng.randint(4, 6)):
                hard_negative(doc)
            labels = doc.labels
            open(path, "w", encoding="utf-8").write("\n".join(doc.lines) + "\n")
        else:
            doc = build_text_doc(rng, n_pos=rng.randint(3, 6))
            labels = doc.labels
            open(path, "w", encoding="utf-8").write("\n".join(doc.lines) + "\n")
        out["documents"].append({"file": f"benchmark/{split}/{name_}", "split": split, "labels": labels})
    with open(os.path.join(ROOT, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    counts = {}
    for d in out["documents"]:
        for lab in d["labels"]:
            counts.setdefault((d["split"], lab["category"]), 0)
            counts[(d["split"], lab["category"])] += 1
    for key in sorted(counts):
        print(key, counts[key])


if __name__ == "__main__":
    main()
