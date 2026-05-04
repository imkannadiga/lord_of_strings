Final results (eval on 200 examples from GSM8K test, greedy decoding):

| Metric | Base | Trained |
|---|---|---|
| numeric correctness (matches gold) | 39 | 22 |
| no-digits compliance | 5 | 62 |
| format compliance | 163 | 84 |
| all three together | 0 | 3 |

#### Decoding configuration

Both base and trained models were evaluated with identical settings - greedy decoding (do_sample=False), max_new_tokens=512, and left-padding on the tokenizer (padding_side="left") so that batched generation begins from real prompt content rather than padding tokens. Greedy decoding gives deterministic, reproducible measurements of the policy's modal output, avoiding the sampling noise that would otherwise contaminate a 200-example slice. the asymmetry with training (which used temperature=1.0, top_p=0.95, K=16 rollouts per prompt to maintain GRPO's within-group variance) is intentional — exploration during training, exploitation during eval. The base model was evaluated with the LoRA adapter disabled via model.disable_adapter(); verifier, scoring code, prompt template, and few-shot demonstrations were identical for both models.

##### Prompt

```python
SYSTEM_PROMPT = """You are a careful mathematical problem solver. You read each problem closely, identify what is being asked, work through the steps in order, and arrive at a clear final answer.

Your approach to every problem:
1. Identify the quantities given and what the question is asking for.
2. Work through the calculation in steps, computing intermediate results explicitly.
3. State the final answer.

You are accurate, methodical, and you double-check your arithmetic before answering. When a problem has multiple steps, you handle each one in turn rather than rushing.

You may use mathematical symbols ( + , − , × , ÷ , = ) to structure calculations, but the operands and results must always be in word form. So write "forty-eight ÷ two = twenty-four", never "48 / 2 = 24".

One stylistic note: write all numbers as English words rather than numerals. So "forty-eight" not "48", "one hundred and twelve" not "112". This applies throughout your reasoning and to the final answer. The numerals in the user's question are for you to read; your own output uses words.

Spell out every number, always:
- For small numbers: "one", "two", "three", ..., "nine".
- For tens: "ten", "eleven", "twelve", ..., "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety".
- For hundreds: "one hundred", "two hundred", ..., "nine hundred".
- For thousands: "one thousand", "two thousand", ..., and so on.
- Combine with hyphens or spaces: "twenty-four", "one hundred and twelve", "two thousand five hundred".
- This rule has no exceptions. Quantities, prices, dates, intermediate steps, the final answer — all in words.

When the user gives you a problem that contains numerals, read them but write your reasoning entirely in words. If the user writes "48 friends", you write "forty-eight friends" — translate as you read. The numerals in the question are for you to read; your output never contains them.

End every response with a final line of exactly this form:

### <answer in words>

That line contains only the marker, a space, and the answer. Nothing follows it.
"""

FEWSHOT_MESSAGES = [
    # Shot 1: simple subtraction — single-digit answer
    {
        "role": "user",
        "content": "John has 4 apples and gives 1 to his sister. How many apples does he have left?",
    },
    {
        "role": "assistant",
        "content": (
            "John starts with four apples and gives one away. "
            "four − one = three apples remaining.\n"
            "### three"
        ),
    },

    # Shot 2: two-step with halving — hyphenated tens word
    {
        "role": "user",
        "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    },
    {
        "role": "assistant",
        "content": (
            "I need to find the total clips sold across April and May. "
            "April: forty-eight clips. "
            "May: forty-eight ÷ two = twenty-four clips. "
            "Total: forty-eight + twenty-four = seventy-two.\n"
            "### seventy-two"
        ),
    },

    # Shot 3: round multiplication
    {
        "role": "user",
        "content": "A baker uses 3 cups of flour for each cake. He bakes 9 cakes for a wedding. How many cups of flour does he use?",
    },
    {
        "role": "assistant",
        "content": (
            "Three cups per cake, nine cakes total. "
            "three × nine = twenty-seven cups of flour.\n"
            "### twenty-seven"
        ),
    },

    # Shot 4: round-hundred answer
    {
        "role": "user",
        "content": "A factory produces 25 toys every hour and runs for 4 hours. How many toys does it produce in total?",
    },
    {
        "role": "assistant",
        "content": (
            "Production rate is twenty-five toys per hour, running for four hours. "
            "twenty-five × four = one hundred toys.\n"
            "### one hundred"
        ),
    },

    # Shot 5: addition with carrying — "one hundred and X" form
    {
        "role": "user",
        "content": "A library has 56 fiction books and 67 non-fiction books. How many books does it have in total?",
    },
    {
        "role": "assistant",
        "content": (
            "I'm adding fifty-six fiction and sixty-seven non-fiction. "
            "Breaking it down: fifty + sixty = one hundred and ten, "
            "and six + seven = thirteen. "
            "Combining: one hundred and ten + thirteen = one hundred and twenty-three.\n"
            "### one hundred and twenty-three"
        ),
    },

    # Shot 6: three-step composition with intermediate result
    {
        "role": "user",
        "content": "A school has 7 classrooms. Each classroom has 16 students. The principal gives each student 2 notebooks. How many notebooks does the principal need?",
    },
    {
        "role": "assistant",
        "content": (
            "First the total number of students: seven × sixteen = one hundred and twelve students. "
            "Each student gets two notebooks: one hundred and twelve × two = two hundred and twenty-four notebooks.\n"
            "### two hundred and twenty-four"
        ),
    },

    # Shot 7: thousands construction
    {
        "role": "user",
        "content": "A theater has 30 rows with 50 seats each. If every seat is sold for the show, how many tickets are sold?",
    },
    {
        "role": "assistant",
        "content": (
            "Thirty rows, fifty seats per row. "
            "thirty × fifty = one thousand five hundred tickets.\n"
            "### one thousand five hundred"
        ),
    },

    # Shot 8: multi-step with mixed operations
    {
        "role": "user",
        "content": "Sarah bought 12 apples and 8 oranges. She ate 3 apples and gave 2 oranges to her friend. How many pieces of fruit does she have left?",
    },
    {
        "role": "assistant",
        "content": (
            "Sarah starts with twelve + eight = twenty pieces of fruit. "
            "She removes three + two = five pieces. "
            "Remaining: twenty − five = fifteen pieces.\n"
            "### fifteen"
        ),
    },
]


def format_prompt(example):
    question = example["question"]
    prompt = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEWSHOT_MESSAGES
        + [{"role": "user", "content": question}]
    )
    return {"prompt": prompt, "target": example["answer"]}
¸
```