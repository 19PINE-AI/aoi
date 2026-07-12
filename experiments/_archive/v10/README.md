# v10 paper drivers

Thin drivers for the v10 ablations referenced in the paper. Each writes
its result JSON into `results/v10_*.json`, which `fill_placeholders.py`
then resolves into the final numbers in `paper/main.tex`.

## Files

- `run_structured.py` — `standard_structured` mode for one CU model on
  the 100 dynamic DynaCU tasks (Section 7.5 / Table 12).
- `run_oss_selection.py` — selection-method ablation on Qwen3-VL-32B
  for the 50 visual-temporal tasks (Section 6.5 / Table 6).
- `run_streaming_sanity.py` — adapter sanity check for Gemini Live and
  OpenAI Realtime on a five-task purely-visual sanity set (Section 7.6 /
  Table 14 / Appendix F).
- `fill_placeholders.py` — reads all v10 result JSONs and substitutes the
  resolved numbers into `paper/main.tex`. Idempotent.

## Re-running the v10 analysis end-to-end

After every new eval JSON lands in `results/`:
```
python experiments/v10/fill_placeholders.py
python experiments/compute_stats.py
python experiments/compute_tokens.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
