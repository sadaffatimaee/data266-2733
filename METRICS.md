SID4=2733, SEED=2733, SLICE=733, HP_ID=3, CLS_A=3, CLS_B=7

Task 1: Groq API, model=openai/gpt-oss-20b, temp=0. Same 2 questions (1 math, 1 logic) across all 6 techniques.

Technique	Calls	Correct	  Notable behavior
Zero-Shot	 1	Yes/Yes	  Bare answer
Few-Shot	 1	Yes/Yes	  Matched example pattern
CoT	         1	Yes/Yes	  Explicit numbered steps
Zero-Shot CoT	 1	Yes/Yes	  Same rigor, no examples needed
Meta-Prompting	 1	Yes/Yes	  Named strategy before solving
Tree of Thoughts 2	Yes/Yes	  3 branches generated + compared

Task 2 Part 1 (unmasked): 51 tokens, vocab 40, embed dim 16. Loss: 2.04 → 0.35 (200 epochs). Attention shape [51,51], clear learned peaks (not random).

Task 2 Part 2 (masked): Separate model, causal mask applied during training. Loss: 2.08 → 0.23. Upper-right triangle of heatmap confirmed zero — no future attention