# What Ollama Is Running

Your app is currently configured to use:

```text
llama3.2:3b
```

From `ollama list`, the installed local model is about:

```text
2.0 GB
```

That 2 GB size does not mean the original model only has 2 GB of full-precision
weights. It usually means Ollama is storing/running a quantized package: the
model has billions of parameters, but the weights are compressed so it can run
locally.

## Ollama's Role

Ollama is not training the model for you. Ollama mainly does this:

1. Downloads a packaged model.
2. Stores it locally.
3. Runs inference locally through a server, usually:

```text
http://localhost:11434
```

4. Lets apps call the model by name:

```bash
ollama run llama3.2:3b
```

## What Your App Sends To Ollama

In this project, the app does not send raw spreadsheet rows to Ollama. It sends
schema-level information and the user request so Ollama can produce a JSON
plan. Then Python validates the plan and pandas computes the actual result.

That matters because the model is not trusted to calculate spreadsheet answers.
It is used as a planner or narrator.

## The Difference Between These Things

Base model:

```text
Llama 3.2 3B
```

Instruction/chat model:

```text
Llama 3.2 3B Instruct
```

Ollama package:

```text
llama3.2:3b
```

Your app behavior:

```text
prompt + schema + validation + pandas execution
```

When you say "make my own LLM," you need to choose which layer you mean.
