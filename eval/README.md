# Synthetic evaluation benchmark

`build_benchmark.py` generates every file here from a fixed seed. All data is synthetic:

- SSNs are random numbers in the valid format, not taken from any record.
- Card numbers are the publicly documented processor test numbers.
- Phone numbers use the fictional 555-01xx range.
- Names are invented.

The labels (`labels.json`) and these rules were committed **before** any detector was written or
tuned, in their own commit. Detectors and prompts may be tuned on the `dev` split only. The
`heldout` split is never used for tuning; `scripts/evaluate.py` reports metrics on it.

## Contents

- 48 documents: 24 `dev` and 24 `heldout`, with 36 `.txt` and 12 `.xlsx` (payroll-like sheets
  plus a notes sheet).
- 6 documents contain only hard negatives, nothing sensitive at all.
- Hard negatives appear throughout:
  - invalid-range SSN-like numbers (area 000, 666 or 900-999; group 00; serial 0000),
  - 9-digit order IDs that pass the ABA checksum,
  - 16-digit tracking numbers that pass the Luhn check,
  - phone numbers, dates, invoice and ticket numbers, ZIP+4 codes, SKUs and prices,
  - HR words in non-HR sentences ("server performance", "terminate the cron job",
    "Account manager: ...").

## Categories

| Type | Category | Meaning |
| --- | --- | --- |
| NPPI | `ssn` | US Social Security number |
| NPPI | `credit_card` | Payment card number |
| NPPI | `routing_number` | ABA routing number |
| NPPI | `bank_account` | Bank account number |
| HRCI | `compensation` | Salary, bonus, offer amounts |
| HRCI | `performance` | PIPs, written warnings, poor ratings |
| HRCI | `termination` | Terminations, severance |
| HRCI | `medical` | Medical leave, workers' compensation claims, disability accommodations |

## Matching rules (fixed before any results were seen)

Each label has a `value`, an exact substring of the document. A prediction is a finding with a
`category` and a `text_snippet`.

1. **Category mapping.** A prediction's category is lowercased and mapped to one of the eight
   categories above by keyword. These are checked in this order, and the first match wins:
   - contains `ssn` or `social` → `ssn`,
   - `card` or `credit` → `credit_card`,
   - `routing` or `aba` → `routing_number`,
   - `account`, `bank` or `iban` → `bank_account`,
   - `salary`, `bonus`, `compensation`, `pay` or `wage` → `compensation`,
   - `performance`, `pip`, `warning` or `review` → `performance`,
   - `termination`, `severance`, `dismiss` or `fired` → `termination`,
   - `medical`, `health`, `leave`, `injury`, `disability` or `workers` → `medical`.

   Anything else is `other`, which never matches a label and counts as a false positive.
2. **Value match.**
   - **NPPI:** a prediction matches a label when the digits of the label value appear in the
     digits of the snippet.
   - **HRCI:** a match needs, after collapsing whitespace and lowercasing, either the label value
     inside the snippet, or the snippet (at least 4 characters) inside the label's sentence.
3. **Counting.** Within one document, each label can be matched once (a true positive). Further
   predictions that match an already matched label are duplicates and are not counted. A
   prediction that matches no label of its category is a false positive. A label that nothing
   matched is a false negative.
4. **Reporting.** Precision, recall and F1 are reported per category and overall (micro),
   summed over the held-out documents.
