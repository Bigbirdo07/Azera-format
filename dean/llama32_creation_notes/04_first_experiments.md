# First Experiments

## Experiment 1: Create A Custom Ollama Wrapper

Create a file named `Modelfile`:

```text
FROM llama3.2:3b

SYSTEM """
You are a dean-office spreadsheet planning assistant.

You never answer directly.
You return one JSON object only.
Use only the exact provided column names.
If the user asks "how many majors are there", use count_unique on Major.
If the user asks "best GPA by major", use groupby_average with group_by Major and value_column GPA.
If the user asks "teacher" or "professor" and no Teacher column exists, use Advisor.
"""

PARAMETER temperature 0.1
```

Create and run:

```bash
ollama create dean-planner-3b -f Modelfile
ollama run dean-planner-3b
```

## Experiment 2: Compare Against Current Model

Ask both models:

```text
how many majors are there?
what is the highest well performing major?
what professor has the best gpa?
how many students in each department?
show the students in the top group
```

Track:

- Did it return JSON?
- Did it use real columns?
- Did it choose the right operation?
- Did it need repair?

## Experiment 3: Build A Training Dataset

Create rows like this:

```json
{
  "user": "how many majors are there?",
  "schema": ["Major", "GPA", "Advisor", "Department"],
  "ideal_plan": {
    "operation": "count_unique",
    "value_column": "Major",
    "filters": []
  }
}
```

Start with 100 examples from real failures and corrected plans.

## Experiment 4: Decide Whether Fine-Tuning Is Needed

Do not fine-tune until prompt examples stop improving results.

Fine-tuning is worth it only if:

- You have many repeated failures.
- You can write correct target JSON.
- You can evaluate before/after accuracy.
- You are comfortable managing model files and adapters.
