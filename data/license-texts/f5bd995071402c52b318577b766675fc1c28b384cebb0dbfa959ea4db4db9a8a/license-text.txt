---
license: apache-2.0
dataset_info:
  features:
  - name: message_id
    dtype: string
  - name: parent_id
    dtype: string
  - name: user_id
    dtype: string
  - name: created_date
    dtype: string
  - name: text
    dtype: string
  - name: role
    dtype: string
  - name: lang
    dtype: string
  - name: review_count
    dtype: int32
  - name: review_result
    dtype: bool
  - name: deleted
    dtype: bool
  - name: rank
    dtype: int32
  - name: synthetic
    dtype: bool
  - name: model_name
    dtype: string
  - name: detoxify
    struct:
    - name: toxicity
      dtype: float64
    - name: severe_toxicity
      dtype: float64
    - name: obscene
      dtype: float64
    - name: identity_attack
      dtype: float64
    - name: insult
      dtype: float64
    - name: threat
      dtype: float64
    - name: sexual_explicit
      dtype: float64
  - name: message_tree_id
    dtype: string
  - name: tree_state
    dtype: string
  - name: emojis
    sequence:
    - name: name
      dtype: string
    - name: count
      dtype: int32
  - name: labels
    sequence:
    - name: name
      dtype: string
    - name: value
      dtype: float64
    - name: count
      dtype: int32
  splits:
  - name: train
    num_bytes: 158850455
    num_examples: 128575
  - name: validation
    num_bytes: 7963122
    num_examples: 6599
  download_size: 66674129
  dataset_size: 166813577
language:
- en
- es
- ru
- de
- pl
- th
- vi
- sv
- bn
- da
- he
- it
- fa
- sk
- id
- nb
- el
- nl
- hu
- eu
- zh
- eo
- ja
- ca
- cs
- bg
- fi
- pt
- tr
- ro
- ar
- uk
- gl
- fr
- ko
tags:
- human-feedback
size_categories:
- 100K<n<1M
pretty_name: OpenAssistant Conversations Release 2
---

# Open Assistant Conversations Dataset Release 2 (OASST2)

## Dataset Description

- **Homepage:** https://www.open-assistant.io/
- **Repository:** https://github.com/LAION-AI/Open-Assistant
- **Paper:** https://arxiv.org/abs/2304.07327

### Dataset Structure

This dataset contains message trees. Each message tree has an initial prompt message as the root node, 
which can have multiple child messages as replies, and these child messages can have multiple replies. 

All messages have a role property: this can either be "assistant" or "prompter". The roles in 
conversation threads from prompt to leaf node strictly alternate between "prompter" and "assistant".

This version of the dataset contains data collected on the [open-assistant.io](https://open-assistant.io/) website until Nov 5 2023.

### JSON Example: Message

For readability, the following JSON examples are shown formatted with indentation on multiple lines.
Objects are stored without indentation (on single lines) in the actual jsonl files.

```json
{
    "message_id": "218440fd-5317-4355-91dc-d001416df62b",
    "parent_id": "13592dfb-a6f9-4748-a92c-32b34e239bb4",
    "user_id": "8e95461f-5e94-4d8b-a2fb-d4717ce973e4",
    "text": "It was the winter of 2035, and artificial intelligence (..)",
    "role": "assistant",
    "lang": "en",
    "review_count": 3,
    "review_result": true,
    "deleted": false,
    "rank": 0,
    "synthetic": true,
    "model_name": "oasst-sft-0_3000,max_new_tokens=400 (..)",
    "labels": {
        "spam": { "value": 0.0, "count": 3 },
        "lang_mismatch": { "value": 0.0, "count": 3 },
        "pii": { "value": 0.0, "count": 3 },
        "not_appropriate": { "value": 0.0, "count": 3 },
        "hate_speech": { "value": 0.0, "count": 3 },
        "sexual_content": { "value": 0.0, "count": 3 },
        "quality": { "value": 0.416, "count": 3 },
        "toxicity": { "value": 0.16, "count": 3 },
        "humor": { "value": 0.0, "count": 3 },
        "creativity": { "value": 0.33, "count": 3 },
        "violence": { "value": 0.16, "count": 3 }
    }
}
```

### JSON Example: Conversation Tree

For readability, only a subset of the message properties is shown here.

```json
{
  "message_tree_id": "14fbb664-a620-45ce-bee4-7c519b16a793",
  "tree_state": "ready_for_export",
  "prompt": {
    "message_id": "14fbb664-a620-45ce-bee4-7c519b16a793",
    "text": "Why can't we divide by 0? (..)",
    "role": "prompter",
    "lang": "en",
    "replies": [
      {
        "message_id": "894d30b6-56b4-4605-a504-89dd15d4d1c8",
        "text": "The reason we cannot divide by zero is because (..)",
        "role": "assistant",
        "lang": "en",
        "replies": [
          // ...
        ]
      },
      {
        "message_id": "84d0913b-0fd9-4508-8ef5-205626a7039d",
        "text": "The reason that the result of a division by zero is (..)",
        "role": "assistant",
        "lang": "en",
        "replies": [
          {
            "message_id": "3352725e-f424-4e3b-a627-b6db831bdbaa",
            "text": "Math is confusing. Like those weird Irrational (..)",
            "role": "prompter",
            "lang": "en",
            "replies": [
              {
                "message_id": "f46207ca-3149-46e9-a466-9163d4ce499c",
                "text": "Irrational numbers are simply numbers (..)",
                "role": "assistant",
                "lang": "en",
                "replies": []
              },
              // ...
            ]
          }
        ]
      }
    ]
  }
}
```

Please refer to [oasst-data](https://github.com/LAION-AI/Open-Assistant/tree/main/oasst-data) for
details about the data structure and Python code to read and write jsonl files containing oasst data objects.


## Main Dataset Files

Conversation data is provided either as nested messages in trees (extension `.trees.jsonl.gz`) 
or as a flat list (table) of messages (extension `.messages.jsonl.gz`).


### Ready For Export Trees

```
2023-11-05_oasst2_ready.trees.jsonl.gz      13,854 trees with 135,174 total messages
2023-11-05_oasst2_ready.messages.jsonl.gz   135,174 messages
```

#### 2023-11-05_oasst2_ready.trees.jsonl.gz Stats
```
Trees            : 13,854
Messages         : 135,174
Oldest message   : 2023-01-16 20:24:26.211711+00:00
Youngest message : 2023-11-04 15:23:03.239343+00:00
Detoxify ratings : 111,448
Accepted messages: 129,517
Deleted messages : 4,376
Tree counts by state:
  - ready_for_export: 13,854
Message counts by language:
  - en: 64,513
  - es: 28,199
  - ru: 13,935
  - zh: 8,615
  - de: 6,145
  - fr: 3,880
  - pt-BR: 2,699
  - th: 1,560
  - ca: 1,283
  - it: 943
  - uk-UA: 845
  - ja: 788
  - pl: 435
  - eo: 295
  - eu: 274
  - vi: 207
  - fi: 138
  - hu: 113
  - ar: 80
  - nl: 72
  - da: 44
  - tr: 37
  - ko: 24
  - he: 24
  - id: 12
  - cs: 12
  - bn: 1
  - sv: 1
```

Trees in ready_for_export state without spam and deleted messages including message labels. The oasst_ready-trees file usually is sufficient for supervised fine-tuning (SFT) & reward model (RM) training.

### All Trees

```
2023-11-05_oasst2_all.trees.jsonl.gz        70,642 trees with 208,584 total messages
2023-11-05_oasst2_all.messages.jsonl.gz     208,584 messages
```

All trees, including those in states prompt_lottery_waiting (trees that consist of only one message, namely the initial prompt), aborted_low_grade (trees that stopped growing because the messages had low quality), and halted_by_moderator.

#### 2023-11-05_oasst2_all.trees.jsonl.gz Stats

```
Trees            : 70,642
Messages         : 208,584
Oldest message   : 2023-01-16 20:24:26.211711+00:00
Youngest message : 2023-11-05 10:24:44.484910+00:00
Detoxify ratings : 156,570
Accepted messages: 189,288
Deleted messages : 5,414
Tree counts by state:
  - ready_for_export: 13,854
  - prompt_lottery_waiting: 44,550
  - halted_by_moderator: 3,089
  - initial_prompt_review: 4,319
  - growing: 3,102
  - aborted_low_grade: 1,708
  - ranking: 20
Message counts by language:
  - en: 85,115
  - es: 47,513
  - ru: 15,990
  - zh: 11,205
  - de: 8,398
  - fr: 5,841
  - pt-BR: 4,540
  - th: 3,236
  - ca: 2,586
  - it: 2,144
  - ja: 1,904
  - uk-UA: 1,889
  - ko: 1,635
  - pl: 1,510
  - eo: 1,405
  - nl: 1,354
  - ar: 1,274
  - vi: 1,137
  - fi: 1,098
  - eu: 995
  - hu: 961
  - tr: 803
  - sv: 763
  - id: 669
  - gl: 574
  - da: 502
  - he: 498
  - cs: 476
  - ro: 434
  - sk: 410
  - fa: 394
  - el: 388
  - bar: 217
  - nb-NO: 196
  - bg: 176
  - bn: 128
  - sl: 119
  - sr: 63
  - swg: 23
  - hi: 14
  - lt: 7
```

### Supplemental Exports: Spam & Prompts 

```
2023-11-05_oasst2_spam.messages.jsonl.gz    19,296 matching messages
```

These are messages which were deleted or have a negative review result ("review_result": false). Besides low quality, a frequent reason for message deletion is a wrong language tag.

```
2023-11-05_oasst2_prompts.messages.jsonl.gz 64,592 matching messages
```

These are all the kept initial prompt messages with positive review result (no spam) of trees in `ready_for_export` or `prompt_lottery_waiting` state.

### Using the Huggingface Datasets

While HF datasets is ideal for tabular datasets, it is not a natural fit for nested data structures like the OpenAssistant conversation trees.
Nevertheless, we make all messages which can also be found in the file `2023-11-05_oasst2_ready.messages.jsonl.gz` available in parquet format as train/validation splits. 
These are directly loadable by [Huggingface Datasets](https://pypi.org/project/datasets/).

To load the oasst2 train & validation splits use:

```python
from datasets import load_dataset
ds = load_dataset("OpenAssistant/oasst2")
train = ds['train']      # len(train)=128575 (95%)
val = ds['validation']   # len(val)=6599 (5%)
```

The messages appear in depth-first order of the message trees.

Full conversation trees can be reconstructed from the flat messages table by using the `parent_id` 
and `message_id` properties to identify the parent-child relationship of messages. The `message_tree_id` 
and `tree_state` properties (only present in flat messages files) can be used to find all messages of a message tree or to select trees by their state.

### Data Visualisation

Explore the content of the prompts from the English subset using [Bunka](https://github.com/charlesdedampierre/BunkaTopics) open-source visualization technology. 
The interactive map [available on a HF space](https://huggingface.co/spaces/bunkalab/visualisation-oasst2) allows to explore each datapoint to get a more precise overview of the contents.

<a href="https://i.imgur.com/B2H8LR3.png">
  <img src="https://i.imgur.com/B2H8LR3.png" alt="Bunka oasst2 Map" width="35%"/>
</a>

## Contact

- Discord [Open Assistant Discord Server](https://ykilcher.com/open-assistant-discord)
- GitHub: [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
- E-Mail: [open-assistant@laion.ai](mailto:open-assistant@laion.ai)
