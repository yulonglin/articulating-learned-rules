# Paper Draft: Can Language Models Learn Rules They Cannot Articulate?

## Status

**Work in Progress - Under Review** (Internal work task, not NeurIPS submission)

## Output

- **PDF**: `main.pdf` (10 pages, 2.8 MB with figures)
- **Source**: `main.tex` (~3,400 words + tables/figures)
- **Figures**: 8 publication-quality figures in `figures/`

## Compilation

```bash
cd paper/
/usr/local/texlive/2025/bin/universal-darwin/pdflatex main.tex
```

Or if `pdflatex` is in PATH:
```bash
cd paper/
pdflatex main.tex
```

## Format

- Double-blind review format (anonymous authors)
- Uses NeurIPS 2025 style for formatting only
- Footer changed from "Submitted to NeurIPS" to "Work in Progress - Under Review"

## Structure

1. **Introduction** - Research question and three-step evaluation pipeline
2. **Related Work** - In-context learning, faithfulness, implicit vs explicit knowledge
3. **Methodology** - Learnability, articulation, and faithfulness testing
4. **Results** - Findings for 31 learnable rules across 4 categories
5. **Discussion** - Main findings, implications, limitations, future directions
6. **Conclusion** - Summary and implications for trustworthy AI

## Key Findings

1. **39% functional-semantic gap**: Models achieve 85-90% functional accuracy but only 50% semantic agreement
2. **Statistical rules show largest gap (58%)**: 89% functional, 31% semantic
3. **70% faithfulness**: Articulations predict counterfactual behavior moderately well
4. **Evidence of post-hoc rationalization**: Some rules show high articulation quality but low faithfulness

## Figures Included

1. **Learnability results** (Fig 1): Overall and category-specific learning curves
2. **Functional vs semantic evaluation** (Fig 2): Judge score vs functional accuracy scatter
3. **Category articulation** (Fig 3): Performance breakdown by rule category
4. **Research Q1 & Q2** (Fig 4): Learnability vs articulation, and articulation vs faithfulness
5. **Research Q3 & Quadrants** (Fig 5): Learnability vs faithfulness, and case study quadrants

## Next Steps

- [ ] Add bibliography/references
- [ ] Add supplementary materials if needed
- [ ] Review and edit for clarity
- [ ] Consider adding error bars/confidence intervals

## Notes

- ✅ **Checklist removed** (not a NeurIPS submission)
- ✅ **Figures added** (8 high-quality PNG files)
- ✅ **Double-blind format** maintained
- `main_draft.tex` was removed (duplicate)
- `neurips_2025.sty` modified to change footer text
- All experiments completed and documented in `../research_log.md`
