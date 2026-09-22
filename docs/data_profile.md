# Data profile

- Rows: **22,465**  |  Columns: 17
- Dates: 2024-01-01 to 2026-02-02
- Companies: JPMORGAN CHASE & CO.
- Channels: {'Web': np.int64(22465)}

## Rows per month

A thin final month is normal -- complaints publish only after the company
responds or 15 days pass, so the tail of any snapshot is partial.

| month   |    n |
|:--------|-----:|
| 2024-01 |  788 |
| 2024-02 |  757 |
| 2024-03 |  854 |
| 2024-04 |  847 |
| 2024-05 |  839 |
| 2024-06 |  802 |
| 2024-07 |  785 |
| 2024-08 |  840 |
| 2024-09 |  810 |
| 2024-10 |  836 |
| 2024-11 |  766 |
| 2024-12 |  895 |
| 2025-01 | 1464 |
| 2025-02 |  883 |
| 2025-03 |  958 |
| 2025-04 |  860 |
| 2025-05 | 1014 |
| 2025-06 |  999 |
| 2025-07 | 1118 |
| 2025-08 |  997 |
| 2025-09 |  943 |
| 2025-10 |  849 |
| 2025-11 |  960 |
| 2025-12 |  744 |
| 2026-01 |  843 |
| 2026-02 |   14 |

## Product distribution

| Product                                                 |    n |   pct |
|:--------------------------------------------------------|-----:|------:|
| Checking or savings account                             | 9299 |  41.4 |
| Credit card                                             | 6384 |  28.4 |
| Credit reporting or other personal consumer reports     | 2987 |  13.3 |
| Money transfer, virtual currency, or money service      | 2028 |   9   |
| Debt collection                                         |  771 |   3.4 |
| Mortgage                                                |  519 |   2.3 |
| Vehicle loan or lease                                   |  377 |   1.7 |
| Debt or credit management                               |   36 |   0.2 |
| Payday loan, title loan, personal loan, or advance loan |   35 |   0.2 |
| Prepaid card                                            |   25 |   0.1 |
| Student loan                                            |    4 |   0   |

## Product x month

Look for a class that stops or starts. That is either a taxonomy change
or a real shift in what gets filed -- both matter for a temporal split.

| month   |   Checking or savings account |   Credit card |   Credit reporting or other personal consumer reports |   Debt collection |   Debt or credit management |   Money transfer, virtual currency, or money service |   Mortgage |   Payday loan, title loan, personal loan, or advance loan |   Prepaid card |   Student loan |   Vehicle loan or lease |
|:--------|------------------------------:|--------------:|------------------------------------------------------:|------------------:|----------------------------:|-----------------------------------------------------:|-----------:|----------------------------------------------------------:|---------------:|---------------:|------------------------:|
| 2024-01 |                           297 |           247 |                                                   112 |                20 |                           0 |                                                   79 |         17 |                                                         4 |              0 |              0 |                      12 |
| 2024-02 |                           287 |           212 |                                                   129 |                19 |                           1 |                                                   80 |         20 |                                                         1 |              1 |              0 |                       7 |
| 2024-03 |                           316 |           278 |                                                   127 |                31 |                           0 |                                                   62 |         21 |                                                         0 |              3 |              0 |                      16 |
| 2024-04 |                           339 |           247 |                                                   138 |                22 |                           1 |                                                   59 |         27 |                                                         0 |              0 |              0 |                      14 |
| 2024-05 |                           358 |           228 |                                                   135 |                21 |                           0 |                                                   61 |         19 |                                                         2 |              1 |              0 |                      14 |
| 2024-06 |                           308 |           267 |                                                   119 |                26 |                           0 |                                                   46 |         16 |                                                         3 |              0 |              0 |                      17 |
| 2024-07 |                           309 |           228 |                                                   112 |                32 |                           1 |                                                   72 |         23 |                                                         0 |              0 |              1 |                       7 |
| 2024-08 |                           324 |           252 |                                                   108 |                27 |                           1 |                                                   77 |         23 |                                                         2 |              1 |              0 |                      25 |
| 2024-09 |                           333 |           267 |                                                   104 |                20 |                           0 |                                                   63 |         16 |                                                         0 |              0 |              0 |                       7 |
| 2024-10 |                           330 |           245 |                                                   114 |                26 |                           1 |                                                   76 |         26 |                                                         1 |              2 |              0 |                      15 |
| 2024-11 |                           287 |           227 |                                                   154 |                16 |                           0 |                                                   56 |         14 |                                                         0 |              0 |              0 |                      12 |
| 2024-12 |                           357 |           253 |                                                   154 |                25 |                           2 |                                                   68 |         20 |                                                         2 |              0 |              0 |                      14 |
| 2025-01 |                           507 |           291 |                                                   227 |                46 |                           3 |                                                  334 |         27 |                                                         0 |              3 |              0 |                      26 |
| 2025-02 |                           360 |           236 |                                                   141 |                23 |                           0 |                                                   84 |         23 |                                                         1 |              2 |              1 |                      12 |
| 2025-03 |                           361 |           260 |                                                   162 |                39 |                           3 |                                                   76 |         24 |                                                         2 |              3 |              2 |                      26 |
| 2025-04 |                           367 |           224 |                                                   129 |                32 |                           3 |                                                   66 |         13 |                                                         3 |              1 |              0 |                      22 |
| 2025-05 |                           471 |           253 |                                                   155 |                25 |                           1 |                                                   80 |         13 |                                                         0 |              1 |              0 |                      15 |
| 2025-06 |                           426 |           288 |                                                   145 |                36 |                           1 |                                                   67 |         20 |                                                         3 |              1 |              0 |                      12 |
| 2025-07 |                           503 |           275 |                                                   146 |                37 |                           2 |                                                  108 |         20 |                                                         0 |              0 |              0 |                      27 |
| 2025-08 |                           439 |           274 |                                                   103 |                45 |                           3 |                                                   99 |         19 |                                                         2 |              0 |              0 |                      13 |
| 2025-09 |                           442 |           297 |                                                    67 |                39 |                           1 |                                                   62 |         26 |                                                         2 |              1 |              0 |                       6 |
| 2025-10 |                           411 |           234 |                                                    69 |                34 |                           3 |                                                   66 |         14 |                                                         1 |              0 |              0 |                      17 |
| 2025-11 |                           426 |           283 |                                                    75 |                64 |                           4 |                                                   79 |         19 |                                                         0 |              1 |              0 |                       9 |
| 2025-12 |                           324 |           238 |                                                    61 |                33 |                           3 |                                                   44 |         23 |                                                         3 |              2 |              0 |                      13 |
| 2026-01 |                           412 |           274 |                                                     1 |                32 |                           2 |                                                   63 |         36 |                                                         3 |              1 |              0 |                      19 |
| 2026-02 |                             5 |             6 |                                                     0 |                 1 |                           0 |                                                    1 |          0 |                                                         0 |              1 |              0 |                       0 |

## Finer labels

- `Sub-product`: 50 distinct
- `Issue`: 76 distinct
- `Sub-issue`: 176 distinct

## Narrative length

|       |   chars |   tokens |
|:------|--------:|---------:|
| count |   22465 |    22465 |
| mean  |    1352 |      237 |
| std   |    1465 |      253 |
| min   |      14 |        2 |
| 50%   |     993 |      175 |
| 90%   |    2751 |      476 |
| 95%   |    3672 |      639 |
| 99%   |    6492 |     1161 |
| max   |   32225 |     5102 |

- Over 128 tokens: **63.4%**
- Over 256 tokens: **31.9%**
- Over 512 tokens: **8.5%**

That last figure sets the truncation length for a transformer. Worth ablating 256 against 512 rather than assuming.

## Redactions

- Contain a redaction: **87.7%**
- Median per narrative: 8  |  max: 1351

## Null rate

|           |   pct_null |
|:----------|-----------:|
| Tags      |       84.2 |
| Sub-issue |        9.6 |
| State     |        1.1 |

**Over 20% null:** Tags — check why before building anything on these.

## Duplicates

- Rows in a repeated group: **356** (1.6%)
- Distinct texts: 22,217

Largest groups:

1. **41 copies** — `i pulled a copy of my consumer report and noticed some items i would like to dispute i have suffered emotional distress as a result of this matter as a result s...`
2. **16 copies** — `the existence of a derogatory rating on my account is causing me significant concern i am deeply worried about its potential impact on my credit it has already ...`
3. **16 copies** — `a collection account is being reported on my credit file that has not been properly validated i exercised my rights under the fair debt collection practices act...`
4. **10 copies** — `this debt collector engaged in abusive deceptive and unfair practices of the fdcpa which it prohibits more so they didn t follow the proper 5 step validation pr...`
5. **8 copies** — `i have formally reported an error regarding incorrect information sent by the credit bureau through my email i am disputing a mistake on my credit card statemen...`

## Leakage check

Columns that describe what happened *after* routing must never become
features. Listed here so the exclusion is deliberate, not accidental.

- `Company response to consumer` — post-hoc outcome, exclude
- `Timely response?` — post-hoc outcome, exclude
