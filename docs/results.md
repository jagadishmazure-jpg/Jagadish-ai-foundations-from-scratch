# Results

Generated from `results/results.json` (a full `aifound demo` run). Do not edit by hand.

<!-- BEGIN:headline -->
| What | Measured (synthetic data, CPU) |
|---|---|
| Hand rules vs learned spam filter | 69.8% vs 90.8% on 500 test messages |
| Gradient descent, house prices | learned $120.8k per 100 m2 and $15.2k per room (true rule 120 / 15); step 0.12 diverges |
| Underfit vs overfit | held-out RMSE 47.3 (degree 0), 12.4 (degree 2), 240.8 (degree 12, train RMSE 3.6) |
| k-means, no labels | purity 1.0 on 150 laundry items |
| More data | spam accuracy 62.0% with 10 examples -> 91.8% with 5000 |
| Neural net depth | 0 hidden layers: 81.6%, 1 hidden layer (32): 99.4%, 2 hidden layers (32, 16): 99.4% |
| Embeddings | same-group cosine 0.997 vs fruit-animal 0.025; analogies 5/5 at 16 dims, 4/5 at 8 |
| Tiny transformer | val loss 3.526 -> 0.259 (1 layer), 0.187 (2 layers) |
| Temperature | distinct outputs of 20: T=0.0: 1, T=0.5: 10, T=1.0: 12, T=1.5: 17 |
| Pretraining -> SFT, held-out places | format 0/6 -> 5/6, correct 0/6 -> 2/6 |
| Diffusion | ring error 0.567 (noise) -> 0.065 after 50 steps (real data 0.016) |
| Multimodal | image->caption 86.0%, caption->image 100.0% (chance 12.5%) |
| Trip planner, $500 | total $294, 4 tool calls, 1 planted ungrounded claim dropped; $150 -> says no |
| Tests | 104 |
<!-- END:headline -->

## Raw output

<!-- BEGIN:results -->
```json
{
  "python": "3.13.5",
  "quick": false,
  "rules_vs_learned": {
    "test_messages": 500,
    "rules_accuracy": 0.698,
    "learned_accuracy": 0.908,
    "top_spam_weights": [
      "prize",
      "click",
      "urgent",
      "cash",
      "free"
    ]
  },
  "gradient_descent": {
    "0.001": {
      "start_mse": 100016.5,
      "final_mse": 1495.7,
      "diverged": false
    },
    "0.01": {
      "start_mse": 100016.5,
      "final_mse": 88.5,
      "diverged": false
    },
    "0.05": {
      "start_mse": 100016.5,
      "final_mse": 88.5,
      "diverged": false
    },
    "0.12": {
      "start_mse": 100016.5,
      "final_mse": null,
      "diverged": true
    },
    "best_fit": {
      "bias": 36.9,
      "per_100m2": 120.8,
      "per_room": 15.2
    },
    "true_rule": {
      "bias": 40,
      "per_100m2": 120,
      "per_room": 15
    }
  },
  "fit_sweep": {
    "train_points": 15,
    "val_points": 300,
    "rows": [
      {
        "degree": 0,
        "train_rmse": 60.1,
        "val_rmse": 47.3
      },
      {
        "degree": 1,
        "train_rmse": 16.0,
        "val_rmse": 18.6
      },
      {
        "degree": 2,
        "train_rmse": 8.0,
        "val_rmse": 12.4
      },
      {
        "degree": 4,
        "train_rmse": 7.6,
        "val_rmse": 13.1
      },
      {
        "degree": 8,
        "train_rmse": 5.8,
        "val_rmse": 37.4
      },
      {
        "degree": 12,
        "train_rmse": 3.6,
        "val_rmse": 240.8
      }
    ],
    "best_degree": 2
  },
  "kmeans": {
    "items": 150,
    "labels_used_for_training": 0,
    "purity_vs_hidden_kind": 1.0
  },
  "learning_curve": [
    {
      "train_examples": 10,
      "test_accuracy": 0.62
    },
    {
      "train_examples": 30,
      "test_accuracy": 0.729
    },
    {
      "train_examples": 100,
      "test_accuracy": 0.9
    },
    {
      "train_examples": 300,
      "test_accuracy": 0.922
    },
    {
      "train_examples": 1000,
      "test_accuracy": 0.922
    },
    {
      "train_examples": 5000,
      "test_accuracy": 0.918
    }
  ],
  "mlp": {
    "train_images": 3000,
    "test_images": 1000,
    "rows": [
      {
        "model": "0 hidden layers",
        "first_epoch_loss": 1.22,
        "last_epoch_loss": 0.357,
        "test_accuracy": 0.816
      },
      {
        "model": "1 hidden layer (32)",
        "first_epoch_loss": 1.246,
        "last_epoch_loss": 0.013,
        "test_accuracy": 0.994
      },
      {
        "model": "2 hidden layers (32, 16)",
        "first_epoch_loss": 1.218,
        "last_epoch_loss": 0.002,
        "test_accuracy": 0.994
      }
    ],
    "example": {
      "true": "x",
      "logits": [
        2.47,
        -3.78,
        -2.18,
        10.0
      ],
      "probabilities": {
        "horizontal": 0.001,
        "vertical": 0.0,
        "box": 0.0,
        "x": 0.999
      }
    }
  },
  "embeddings": {
    "vocab": 58,
    "dims": 16,
    "mean_cos_same_group": 0.997,
    "mean_cos_fruit_vs_animal": 0.025,
    "analogies": [
      {
        "query": "king - man + woman",
        "expected": "queen",
        "got": "queen"
      },
      {
        "query": "boy - man + woman",
        "expected": "girl",
        "got": "girl"
      },
      {
        "query": "queen - king + prince",
        "expected": "princess",
        "got": "princess"
      },
      {
        "query": "girl - boy + father",
        "expected": "mother",
        "got": "mother"
      },
      {
        "query": "uncle - man + woman",
        "expected": "aunt",
        "got": "aunt"
      }
    ],
    "analogy_hits": "5/5",
    "analogy_hits_at_8_dims": "4/5"
  },
  "attention_toy": {
    "tokens": [
      "the",
      "cat",
      "sat",
      "because",
      "it"
    ],
    "weights_from_it": {
      "the": 0.04,
      "cat": 0.812,
      "sat": 0.04,
      "because": 0.04,
      "it": 0.067
    }
  },
  "tiny_lm": {
    "corpus_chars": 98155,
    "vocab_chars": 30,
    "pretrain_steps": 600,
    "sft_steps": 300,
    "depth": [
      {
        "layers": 1,
        "params": 53726,
        "val_loss_before": 3.526,
        "val_loss_after": 0.259,
        "train_loss_last": 0.26
      },
      {
        "layers": 2,
        "params": 99358,
        "val_loss_before": 3.53,
        "val_loss_after": 0.187,
        "train_loss_last": 0.195
      }
    ],
    "temperature": [
      {
        "temperature": 0.0,
        "distinct_of_20": 1,
        "next_char_entropy_bits": 0.0,
        "sample": "ooty for the gardens. pe"
      },
      {
        "temperature": 0.5,
        "distinct_of_20": 10,
        "next_char_entropy_bits": 2.754,
        "sample": "leh for the passes. peop"
      },
      {
        "temperature": 1.0,
        "distinct_of_20": 12,
        "next_char_entropy_bits": 2.967,
        "sample": "leh for the passes. peop"
      },
      {
        "temperature": 1.5,
        "distinct_of_20": 17,
        "next_char_entropy_bits": 3.169,
        "sample": "leh for the passes. peop"
      }
    ],
    "heldout_places": [
      "goa",
      "leh",
      "agra"
    ],
    "qa_after_pretraining": {
      "format_ok": "0/6",
      "correct": "0/6",
      "cases": [
        {
          "prompt": "Q: weather in goa? A:",
          "answer": "the weather is warm. people v"
        },
        {
          "prompt": "Q: goa is known for? A:",
          "answer": "h tombs. people visit shimla f"
        },
        {
          "prompt": "Q: weather in leh? A:",
          "answer": "has passes. people visit pune"
        },
        {
          "prompt": "Q: leh is known for? A:",
          "answer": "hills. people visit pune for t"
        },
        {
          "prompt": "Q: weather in agra? A:",
          "answer": "the weather is hot. people vi"
        },
        {
          "prompt": "Q: agra is known for? A:",
          "answer": "ate weather is cold. people vi"
        }
      ]
    },
    "qa_after_sft": {
      "format_ok": "5/6",
      "correct": "2/6",
      "cases": [
        {
          "prompt": "Q: weather in goa? A:",
          "answer": "the weather is warm. people v"
        },
        {
          "prompt": "Q: goa is known for? A:",
          "answer": "beaches."
        },
        {
          "prompt": "Q: weather in leh? A:",
          "answer": "cold."
        },
        {
          "prompt": "Q: leh is known for? A:",
          "answer": "cafes."
        },
        {
          "prompt": "Q: weather in agra? A:",
          "answer": "thes."
        },
        {
          "prompt": "Q: agra is known for? A:",
          "answer": "thers."
        }
      ]
    }
  },
  "diffusion": {
    "train_steps": 8000,
    "denoise_steps": 50,
    "samples": 1000,
    "ring_error_by_step": {
      "0": 0.567,
      "10": 0.566,
      "25": 0.506,
      "40": 0.32,
      "50": 0.065
    },
    "ring_error_real_data": 0.016
  },
  "multimodal": {
    "train_steps": 1000,
    "test_images": 500,
    "captions": 8,
    "chance": 0.125,
    "image_to_caption_top1": 0.86,
    "caption_to_image_top1": 1.0,
    "side_right": 1.0,
    "shape_right": 0.86
  },
  "trip": {
    "under_500": {
      "feasible": true,
      "total": 294,
      "answer": [
        "goa dry season runs from november to february with calm sea and warm days [kb-goa-season]",
        "scooter rental is the cheapest way to move between goa beaches [kb-goa-transport]",
        "forecast high 31 C with 2 mm rain [tool:get_weather]",
        "Breeze Rail round trip $62 [tool:get_fare]",
        "Palmrest Inn (south goa) at $38/night [tool:search_stays]",
        "total $294 for 4 nights incl. $20/day food [tool:calculator]"
      ],
      "dropped_ungrounded": [
        "the beach is free of any entry fee all year"
      ],
      "tool_calls": 4,
      "steps": 8,
      "request": "plan my goa trip under $500, mail me at [EMAIL]"
    },
    "under_150": {
      "feasible": false,
      "total": null,
      "answer": [
        "no plan fits this budget with the stays on offer; raise the budget or cut nights"
      ],
      "tool_calls": 3
    },
    "context_window": {
      "turns": 31,
      "budget_words": 120,
      "kept_plain": 13,
      "name_survives_plain": false,
      "kept_pinned": 13,
      "name_survives_pinned": true
    }
  },
  "runtime_seconds": 28.9
}
```
<!-- END:results -->
