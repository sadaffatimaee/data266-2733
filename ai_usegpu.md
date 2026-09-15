

Which parts did you use an assistant for?

I used Claude AI for coding all benchmark templates, understanding GPU concepts, and debugging.
I wrote myself the OOM boundary testing and result interpretation.

One specific thing wrong: Initial Part E didn't handle nvidia-smi failures.
How I found it: Part E returned None values for metrics.
What I changed: Added try-except and timeout handling for robustness.
