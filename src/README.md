Pipeline: start by brainstorming rule candidates with src/brainstorm_rules.py (LLM-generated, category-balanced rules), then winnow them using src/curate_rules.py to keep diverse,
  implementable options. Build balanced labeled datasets per rule via src/generate_datasets.py, leaning on programmatic generators whenever possible. Validate Step 1 learnability with
  src/test_learnability.py (few-shot prompts, no CoT) and summarize with src/analyze_learnability.py to filter for >90 % accuracy rules.

  For Step 2 articulation, probe surviving rules with src/test_articulation_mc.py (multiple choice) and src/test_articulation_freeform.py (free-form descriptions plus LLM-judge/
  functional scoring), then aggregate using src/analyze_articulation_freeform.py.

  Step 3 faithfulness runs through src/test_faithfulness.py, which generates counterfactuals, functional rechecks, and cross-context probes; draw out the findings with downstream
  analysis scripts.

  Throughout, src/runner.py plus src/api_caller.py handle reproducible experiment execution and caching. When it’s time to present results, src/analyze.py offers quick diagnostics and
  src/create_visualizations.py produces the figures the spec calls for.
  