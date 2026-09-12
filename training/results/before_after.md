# What training changed

Same model, same prompt, same validation example - before and after fine-tuning
`google/flan-t5-base` on 1,020 examples generated from our own finding logic.

Run on 12 September 2026, Colab T4, 6 epochs, about 9 minutes of training.

---

## Example 1 - a recurring issue

**Input the model was given** (produced by our deterministic engine, not by a person):

```
finding: recurring_issue | company: Initech Corp | issue: rising leverage |
years: 2022,2023,2024 | consecutive_periods: 3
```

**Before training**

> Initech Corp. has a recurring issue with rising leverage in the years 2022, 2023, 2024 and 2023.

The base model restates the input, repeats 2023, and says nothing a reviewer could act on.

**After training**

> Rising leverage has persisted across 3 consecutive reporting periods (FY2022 to FY2024). A sustained trend of this kind is more significant than any single year's movement and should be addressed in management commentary.

**What a reviewer would have written**

> Rising leverage has persisted across 3 consecutive reporting periods (FY2022 to FY2024). A sustained trend of this kind is more significant than any single year's movement and should be addressed in management commentary.

Word for word. The model learned the reporting convention - periods rather than
a list of years, the "FY" prefix, and the judgement that a repeated finding
outweighs a single year.

---

## Example 2 - a trend across five years

```
finding: recurring_issue | company: Initech Corp | issue: declining operating margin |
years: 2017,2018,2019,2020,2021 | consecutive_periods: 5
```

**After training**

> Declining operating margin has persisted across 5 consecutive reporting periods (FY2017 to FY2021). A sustained trend of this kind is more significant than any single year's movement and should be addressed in management commentary.

Again matching the reference, on a different issue and a different span of
years, so it has learned the mapping rather than memorised one sentence.

---

## Example 3 - where it gets a number wrong, and why that matters

```
finding: arithmetic_inconsistency | company: Cobalt Mining | year: 2022 |
rule: Gross Profit = Revenue - Cost of Revenue |
revenue: 105150 | cost of revenue: 74670 | computed: 30480 | reported: 56210
```

**After training**

> Gross Profit for FY2022 does not follow from its components. With revenue of USD 105,150 and cost of revenue of USD **75,670**, the figure should be USD 30,480, but USD 56,210 is reported. The line appears overstated by USD 27,730.

The prose is right, the conclusion is right, and the arithmetic it quotes is
right - but it typed the cost of revenue as **75,670** when the input said
**74,670**.

This is the most useful result of the whole exercise, and it belongs in the
presentation rather than being hidden. It is a 250M-parameter model; it can
express a finding, and it cannot be trusted to carry a figure. That is exactly
why the architecture never lets it compute or quote one unchecked:

- every figure a reviewer sees is computed by the deterministic engine
- the Evidence Agent passes the model only figures already established as true
- the runtime groundedness check refuses any number in the narrative that is
  not in the evidence

A model that never made this mistake in 180 examples would prove nothing, since
we could not tell whether the safeguards worked. This one shows they are needed.
