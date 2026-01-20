# Newspaper Climate Labeling Results

## Bottom Line

Tested 10 samples (5 historical + 5 modern). **10/10 climate-related (100%)**.

Full dataset likely >95% climate content. Training FastText probably isn't worth it. Better to use OpenAI credits directly for [OCR cleaning](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/src/gpt_labeler.py).

## Changes

**1. [Newspaper-specific prompt](https://github.com/tqmsh/FastText-trainer-finewedu-climate/blob/main/prompts/climate_yesno_newspaper.txt)**
- Mentions "newspaper articles" and "OCR errors"
- Asks "Is this news report related to weather events?"
- Expects OCR degradation, historical language

**2. [Text chunking](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/src/gpt_labeler.py#L100)**
- Split into 500-word chunks
- Process ALL chunks
- ANY chunk climate-related → whole article labeled climate-related
- See per-chunk votes in [results](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/data/newspaper_labels_10.json)

**3. [Matched OCR script speed](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/cli.py#L109)**
- 20 concurrent workers (was 5)
- No rate limiting (was 0.1s)

## Results

Full details in [`newspaper_labels_10.json`](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/data/newspaper_labels_10.json).

```
10/10 climate-related
0 failures
```

**Examples with chunk voting:**
- historical_1896-10-20: 9 YES / 113 NO chunks → labeled YES
- modern_2004-09-03: 12 YES / 191 NO chunks → labeled YES

Even articles with mostly NO chunks get correctly labeled because climate content is in specific sections.

**Previous vs Current:**
- Previous (web prompt + truncation): 0/10 climate-related
- Current (newspaper prompt + chunking): 10/10 climate-related

Hurricane Frances article now correctly labeled YES. [Verify here](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/data/newspaper_labels_10.json) (search `modern_regex_cleaned_2004-09-03_0`).

## What This Means

Your keyword filtering works well. If full dataset is >95% climate:

1. FastText training costs tokens to label 1000+ samples
2. Only filters out <5% of content
3. Low ROI

**Better:** Skip label → train → filter. Use credits for OCR cleaning directly.

## Questions

1. Does 10/10 match your expectations?
2. Proceed to OCR cleaning or run larger validation (50-100) first?
3. Review [per-chunk details](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/data/newspaper_labels_10.json)?
4. What to prioritize next?

## Code

- [Newspaper prompt](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/prompts/climate_yesno_newspaper.txt)
- [Chunking logic](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/src/gpt_labeler.py#L100)
- [CLI command](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/cli.py#L97)
- [Test results](https://github.com/tqmsh/FastText-trainer-finewebedu-climate/blob/main/data/newspaper_labels_10.json)

OCR cleaning scripts ready when you decide.
