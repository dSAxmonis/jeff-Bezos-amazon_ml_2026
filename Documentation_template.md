# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
The pipeline has five parts:
1. **Country-agnostic normalization.** This includes a transliteration dictionary learned from the training pairs and an address parser that splits out components.
2. **Multi-view sparse blocking.** Name, name+address and address-only TF-IDF, a reverse pass and exact keys. Recall on held-out S1 is 99.07%.
3. **A streaming prune model** that reduces this to ≤20 candidates per S1.
4. **Two LightGBM stages.** The second models how candidates interact: competition between S1 entities for the same record, S2↔S3 agreement, same-name twins and co-located businesses.
5. **A decision rule** that assigns each S2/S3 record to at most one S1, then chooses, per S1 entity, the subset of candidates that maximizes expected F0.5.

Out-of-fold macro F0.5 on 1.77M held-out training S1 entities is **0.98654**.

---

## 2. Methodology

### 2.1 Problem Analysis
All findings below were measured on the full training data:
- **Every S2/S3 record matches at most one S1** (7.64M GT pairs, 7.64M unique IDs). This makes the task an assignment problem.
- **Names are reused.** 51% of S1 entities share their suffix-stripped name with another S1 entity in a different location. Among US pairs with the same name key, 0.6M are true and 18.9M are false. The address is the primary discriminator.
- **Distractors are adversarial twins.** About 30% of unmatched S2/S3 records share a name with an S1 entity, and about 70% of singletons have such a lookalike. The hardest ones sit on the same street with a nearby house number. True matches also carry number noise: leading zeros, truncation, ±1, and cases where only S2 and S3 agree with each other.
- **Indic-script names are word-by-word renderings of English words.** Token counts align in 100% of pairs, so a dictionary of about 1.35K entries learned from training pairs covers about 96% of test Indic tokens.
- Other noise: domains as names (`.com` only), alias markers (`dba`, `d.b.a`, `aka`, `t/a`, `trading as`, `M/s`), junk prefixes, leet digits, accents, component reordering, `null`, empty addresses (~3.5%), and junk trade names that can only be matched by address.
- **Address label conflicts.** S2/S3 label Telangana cities as Andhra Pradesh (29K training pairs). "Washington" is both a state name and the DC city.
- **France (15% of test) has no training data.** Its addresses concentrate in ~18 cities, and its names use a generic vocabulary. Test also has more S2/S3 records per S1 (5.75 vs 4.68), which suggests more distractors.

### 2.2 Solution Strategy
**Approach type:** blocking, then a learned pairwise classifier, then a learned context (graph) model, then a structured decision rule.
**Core innovations:**
1. Address-first, country-agnostic features, including house-number relations and name- and address-ambiguity counts.
2. A context model that encodes one-to-one competition, S2↔S3 agreement and twin detection.
3. Per-entity expected-F0.5 subset selection on calibrated probabilities, instead of a global threshold.
4. A hidden-S1 validation protocol (20% of train S1 never queried) that simulates the orphan and distractor rate expected in test.

---

## 3. Candidate Generation (Blocking)
- **Normalization first.** Names become core tokens (legal suffixes canonicalized and removed), a sorted name key, alias, leet and domain variants, and transliteration. Addresses become state/region, localities, postcode, house number (plus suffix), unit, and expanded street tokens. Parsing is component-based, so `ct` (court vs Connecticut) and `st` (street vs saint) don't collide.
- **Partitions:** (country, state). Records with no state are matched in a separate pass against the whole country, and a missing state is filled from a city→state map learned on S1.
- **Sparse TF-IDF top-k** (sparse_dot_topn), within each partition:
  - char 3-grams on the name, top 30;
  - char 3-grams on name + street + city, top 30;
  - word TF-IDF on address tokens + numbers + house number, top 30;
  - n-grams present in more than 2% of documents are dropped (a 20× speed-up);
  - a reverse pass takes each pool record's top 5 S1 entities (top 20 by name when the record has no address).
- **Exact keys:** (locality, street), (state, name key), (country, name key) for records without a state, and (locality, house number, first street token). Blocks larger than 300 are skipped.
- **Candidate pairs:** the union is 225.7M train and 211.9M test pairs (about 100 per S1), with recall 99.07%. A streaming LightGBM prune model (25 cheap features, out-of-fold on train) keeps the top 20 per S1 with p ≥ 0.005. That leaves 7.8 (train) / 8.9 (test) per S1 at recall 98.64%. **This pruned set is `candidate_pairs.tsv`.**
- **Keeping true matches:** we measured the recall of each blocker alone and added blockers aimed at the missed buckets: address-only for junk names, reverse-by-name for empty addresses, and the state-conflict fixes.

---

## 4. Matching Model
**Features (~95 pairwise + ~25 context):**
- **Name:** token-set, token-sort, partial and plain ratio, Jaro-Winkler, Levenshtein on concatenated names (for domains), a consonant skeleton (for transliteration), token Jaccard and containment in both directions, IDF-weighted overlap, the IDF of the rarest unmatched token on each side, best similarity over alias/leet/domain variants, legal-suffix relation, and script/domain/alias flags.
- **Ambiguity:** how many S1 entities share the name key (in the country and in the city), how many pool records share it, and how many S1 entities share the exact address or the (city, house number). The last two cover co-located businesses.
- **Address:** house number equal (raw and zero-stripped), prefix relation, absolute and relative difference, digit edit distance, suffix (bis/ter/letter), unit, postcode, city, state, street token overlap and fuzzy scores, all-token IDF overlap, and landmark flags.
- **Blocking:** scores and ranks from each blocker, provenance flags, the prune score and rank.
- **Context (stage 2):**
  - within S1: rank, gap to best, share, expected cluster size;
  - competition: the same candidate's best score for another S1, and the margin to it;
  - agreement: similarity to the S1's other top-3 candidates weighted by their probability, including an S2↔S3-only version;
  - twins: a same-name candidate with a different house number that scores higher.

**Model type:** LightGBM (MIT). Stage 1 is pairwise and stage 2 is the context model. Both use 5-fold GroupKFold by S1 with out-of-fold predictions, and stage 2 is calibrated with isotonic regression.
An optional multilingual cross-encoder (`intfloat/multilingual-e5-small`, MIT, 118M parameters) is implemented. It is trained on the hidden S1 set and scores the ambiguous band as an extra stage-2 feature.

**Threshold selection:**
1. One-to-one assignment: each S2/S3 record goes to its best S1 only, with margin δ.
2. For each S1, choose the top-k that maximizes expected F0.5, estimated by Monte-Carlo over Bernoulli(p). Here k = 0 scores P(no match), which handles singletons.
3. The calibration adjustment (a, b) and δ are grid-searched on out-of-fold predictions. A tuned global threshold was compared and loses (0.98635 vs 0.98654).

---

## 5. Results & Error Analysis
- **F0.5 (macro, out-of-fold, 1.77M S1, 20% of S1 hidden as orphans):** 0.98654. By country: US 0.98731, India 0.98537. Singletons 0.98733, matched 0.98649.
- **Micro precision / recall:** 0.9977 / 0.9645. On test, 1,628,103 of 1,732,544 S1 entities receive at least one match (5.63M matches). France gets 3.13 matches per S1, in line with US (3.35) and India (3.20).
- **Ablation on the full out-of-fold set (1.77M S1):**

| System | Macro F0.5 |
|---|---|
| Stage-1 pairwise model, threshold 0.5 | 0.98187 |
| + one-to-one assignment and per-S1 expected-F0.5 | 0.98437 |
| Stage-2 context model, threshold 0.5 | 0.98542 |
| Stage-2, best global threshold after one-to-one (t=0.7) | 0.98635 |
| **Stage-2 + one-to-one (δ=0.1) + expected-F0.5 (final)** | **0.98654** |

- **Development-slice ablations** (a 180K-S1 subset for fast iteration): adding co-located-address features moved 0.98876 to 0.98906, and singleton F0.5 from 0.9827 to 0.9847.
- **Common false positives:**
  - same-name records at a nearby house number (twins);
  - junk-named records at an address shared by two S1 businesses;
  - records belonging to an S1 entity absent from the query set (orphans).
- **Common false negatives:**
  - candidates with an empty address and a reused name (genuinely ambiguous);
  - junk trade names with sparse addresses;
  - heavy number noise on both the name and the address.

---

## 6. Conclusion
The address is the key signal on this data. Names are too heavily reused to identify a business on their own. The largest gains came from recognizing the structure of the data: one-to-one assignment, adversarial twins, S2↔S3 agreement, and per-entity F0.5 optimization. Model complexity mattered less. The whole pipeline is CPU-based and country-agnostic, and it applies to France without French training data.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`. The entry point is `run_pipeline.py`, which runs io, splits, normalize, sets, blocking, prune, stage1, context and output, and writes both TSVs to `output/` after the official validator passes. See `README.md` for commands, runtimes, the optional GPU cross-encoder and the leaderboard-probe helper.

### B. Additional Results
| Stage | Output | Wall-clock (M3 Max, 16 cores / 48 GB) |
|---|---|---|
| Normalization (24.2M records) | parsed names and addresses | 2 min |
| Blocking | 225.7M train / 211.9M test pairs; Q recall 99.07% | 16 + 12 min |
| Prune (streaming LightGBM) | 17.3M train / 15.4M test pairs; Q recall 98.64% | ~100 min |
| Stage-1 LightGBM (93 features, 5 folds) | OOF p1 | 58 min |
| Context + stage-2 (115 features) + decision tuning | OOF p2, F0.5 0.98654 | 19 min |
| Output + validator | both TSVs, PASS | 20 s |

Blocking recall of each blocker alone, on the query set: name 0.780, name+address 0.883, address-only 0.893, reverse 0.923, exact keys 0.897. The union reaches 0.991.

Top stage-1 features by gain: prune score, prune rank, house-number absolute difference, IDF of the rarest unmatched candidate name token, legal-suffix relation, token-sort ratio, name-key frequency among S1, and number-set Jaccard.

Optional cross-encoder inputs are prepared (2.59M H training pairs; scoring bands of 7.4M train and 7.4M test pairs). When its scores are present they enter stage 2 as extra features (`ce`, `ce_rank`, `ce_gap`, `ce_cgap`).
