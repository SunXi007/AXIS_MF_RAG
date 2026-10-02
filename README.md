# Axis MF Facts Chatbot

A retrieval-augmented (RAG) assistant that answers **factual** questions about a
fixed set of Axis Mutual Fund schemes. Every answer is quoted from official
Axis Mutual Fund pages and statutory documents, with the source linked. If the
documents do not cover something, the bot says so instead of guessing.

It does **not** give investment advice, compare returns, or predict performance.

---

## Source list

Only official Axis Mutual Fund (`axismf.com` / `transact.axismf.com`) sources are
indexed. Third-party aggregators are deliberately excluded.

| ID | Scheme | Plan | Document | Link |
|---|---|---|---|---|
| s01 | Axis Large Cap Fund | Direct | Scheme page | [open](https://www.axismf.com/mutual-funds/equity-funds/axis-large-cap-fund/ef-dg/direct) |
| s02 | Axis Large Cap Fund | Regular | Scheme page | [open](https://www.axismf.com/mutual-funds/equity-funds/axis-large-cap-fund/ef-gp/regular) |
| s03 | Axis Large Cap Fund | All plans | SID | [open](https://www.axismf.com/cms/sites/default/files/Statutory/Axis%20Bluechip%20Fund%20-%20SID.pdf) |
| s04 | Axis Flexi Cap Fund | Direct | Scheme page | [open](https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/ml-dg/direct) |
| s05 | Axis Flexi Cap Fund | Regular | Scheme page | [open](https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/ml-gp/regular) |
| s06 | Axis Flexi Cap Fund | All plans | SID | [open](https://transact.axismf.com/cms/sites/default/files/Statutory/Axis%20Flexi%20Cap%20Fund%20-%20SID.pdf) |
| s07 | Axis Flexi Cap Fund | All plans | Factsheet (Nov 2024) | [open](https://www.axismf.com/cms/sites/default/files/pdf-factsheets/20190204018-Flexi%20Cap%20Fund%20(November%202024)%20DP-Leaflet.pdf) |
| s08 | Axis Flexi Cap Fund | All plans | Factsheet | [open](https://transact.axismf.com/cms/sites/default/files/pdf-factsheets/Axis%20Flexi%20Cap.pdf) |
| s09 | Axis ELSS Tax Saver Fund | Direct | Scheme page | [open](https://www.axismf.com/mutual-funds/equity-funds/axis-elss-tax-saver-fund/ts-dg/direct) |
| s10 | Axis ELSS Tax Saver Fund | Regular | Scheme page | [open](https://www.axismf.com/mutual-funds/equity-funds/axis-elss-tax-saver-fund/ts-gp/regular) |
| s11 | Axis ELSS Tax Saver Fund | All plans | KIM + application form | [open](https://www.axismf.com/1/5/464/2258/4302/kim_and_application_form_axis_elss_tax_saver_fund.pdf) |
| s12 | Axis ELSS Tax Saver Fund | All plans | SID | [open](https://www.axismf.com/cms/sites/default/files/Statutory/Axis%20ELSS%20Tax%20Saver%20Fund%20-%20SID.pdf) |
| s13 | Axis ELSS Tax Saver Fund | All plans | E-factsheet (Jan 2025) | [open](https://www.axismf.com/efactsheet/January-2025/Innerpage/ELSS-TAX-SAVER-FUND.html) |
| s14 | Axis Midcap Fund | Direct | Scheme page | [open](https://www.axismf.com/mutual-funds/equity-funds/axis-mid-cap-fund/mc-dg/direct) |
| s16 | Axis Mutual Fund (AMC) | — | Homepage | [open](https://www.axismf.com/) |
| s17 | Axis Mutual Fund (AMC) | — | Downloads index | [open](https://www.axismf.com/downloads) |
| s18 | Axis Mutual Fund (AMC) | — | Factsheet (Jul 2026) | [open](https://www.axismf.com/1/5/1423/1484/1487/2872/4561/Axis_Fund_Factsheet_July_2026_2412c4ee93.pdf) |
| s19 | Axis Mutual Fund (AMC) | — | Factsheet (Mar 2026) | [open](https://transact.axismf.com/cms/sites/default/files/pdf-factsheets/Axis%20Fund%20Factsheet%20March%202026.pdf) |

> The full machine-readable list, including the excluded third-party entry
> (`s15`, INDmoney) and the reason for its exclusion, is in
> [`sources.csv`](sources.csv).

**Indexed coverage:** Axis Large Cap Fund, Axis Flexi Cap Fund, Axis ELSS Tax
Saver Fund, Axis Midcap Fund, plus AMC-wide factsheets, holdings and downloads.
Schemes outside this set (e.g. Axis Small Cap Fund) are not indexed and return a
"not found" style answer rather than a guess.

---

## Sample Q&A

Answers below are illustrative: factual replies are generated per question and
always cite the chunk and source they came from, so wording varies slightly.
Refusals are fixed templates.

**Grounded, factual**

> **Q:** What is the lock-in period of the Axis ELSS Tax Saver Fund?
>
> **A:** The Axis ELSS Tax Saver Fund carries a statutory lock-in of three years
> from the date of allotment. *Source: Axis ELSS Tax Saver Fund — SID.*

> **Q:** What is the exit load on the Axis Large Cap Fund Regular plan?
>
> **A:** There is no exit load (Exit Load — NIL) on the Axis Large Cap Fund.
> *Source: Axis Large Cap Fund — Regular plan scheme page.*

> **Q:** What is the minimum lump sum investment for the Axis ELSS Tax Saver Fund?
>
> **A:** The minimum lump sum investment is ₹10,000, for both the Regular and
> Direct plans. *Source: Axis ELSS Tax Saver Fund — SID.*

**Refused: investment advice**

> **Q:** Should I invest in the Axis ELSS Tax Saver Fund?
>
> **A:** I can share facts about these schemes, but I don't give investment
> advice. For guidance on choosing a scheme, please speak with a SEBI-registered
> investment adviser. You can review scheme facts here:
> https://www.axismf.com/downloads

**Refused: performance / returns**

> **Q:** Which Axis fund gave the best returns last year?
>
> **A:** I don't calculate or compare returns. The official monthly factsheet has
> the official performance figures:
> https://www.axismf.com/1/5/1423/1484/1487/2872/4561/Axis_Fund_Factsheet_July_2026_2412c4ee93.pdf

**Not covered by the sources**

> **Q:** What is the expense ratio of the Axis Small Cap Fund?
>
> **A:** I couldn't find that in the official sources I'm using (Axis Mutual
> Fund — Large Cap, Flexi Cap, ELSS Tax Saver, Midcap). Try asking about expense
> ratio, exit load, minimum SIP, ELSS lock-in, benchmark, riskometer, or how to
> download a statement: https://www.axismf.com/downloads

**Personal identifiers blocked**

> **Q:** My PAN is ABCDE1234F — what is the exit load?
>
> **A:** For your security I can't process personal identifiers. Please remove
> account numbers, PAN, Aadhaar, OTPs, email addresses, or phone numbers.
> — Facts-only. No investment advice.

**Off-topic**

> **Q:** How do I cook pasta?
>
> **A:** I don't have that in the sources I use. — Facts-only. No investment
> advice.

---

## Disclaimer

> Facts-only. No investment advice.
> Answers are generated only from official Axis Mutual Fund public pages and
> statutory documents.
> Mutual fund investments are subject to market risks. Read all scheme related
> documents carefully.

This project is an independent, non-commercial demonstration. It is not
affiliated with, endorsed by, or operated by Axis Mutual Fund or INDmoney, and
nothing it produces is investment advice. Always refer to the official scheme
documents before making any decision.
