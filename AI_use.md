Used Claude to scaffold the LangChain prompting code and the from-scratch PyTorch attention implementation. I picked the questions based on the criteria , ran every cell, and debugged.

Claude suggested two Llama model names — both threw `NotFoundError: 404, model does not exist or you do not have access to it.

Found out by running the cell — got the 404 traceback both times.
 
Fixed it by querying https://api.groq.com/openai/v1/models to see what my account could actually access, then switched to openai/gpt-oss-20b. Works because it's based on real account permissions, not a guessed model name.