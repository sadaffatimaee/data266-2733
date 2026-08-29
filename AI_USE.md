

 1. Which parts did you use an assistant for, and which did you write yourself?
I'm new to CUDA and PyTorch, so I used AI to help me understand the basic skeleton of things like tensors, CUDA kernels, and blocks/threads — the syntax was hard to follow at first. AI also helped me debug compile errors and structure these summary files (METRICS.md, RUN_LOG.txt, this file).

2. Give one specific thing it produced that was wrong. Paste the wrong output.
While I was recovering from a Colab "Review remote changes" save conflict, AI told me to click "Save your changes." That was wrong — it would have overwritten my completed N=4096 CUDA result with an incomplete local version.
Wrong output given: Click 'Save your changes.

3. How did you find out? What did the failure look like?
I sent a screenshot of the actual dialog, which clearly labeled the two options: "Remote changes" (left) and "Your unsaved changes" (right). Looking at those labels again, it was clear my completed N=4096 data was in "Remote changes," not in my local unsaved copy so the advice to save my local changes would have destroyed the good data.

4. What did you change, and why does your version work?
AI corrected itself and told me to click "Cancel" instead, then reload the notebook. I did this and confirmed my completed N=4096 result was preserved. This worked because "Cancel" discards only the unsaved local edits and keeps the version already saved to Colab's servers, which was the one containing my good data.