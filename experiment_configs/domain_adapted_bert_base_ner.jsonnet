{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "IOB1",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
     "bert": {
         "type": "bert-pretrained",
         "pretrained_model": "models/conll2003_domain_adapted_bert_base_cased/"
     }
    }
  },
  "train_data_path": "data/conll2003_ner/eng.train",
  "validation_data_path": "data/conll2003_ner/eng.testa",
  "model": {
    "type": "simple_tagger",
    "calculate_span_f1": true,
    "label_encoding": "IOB1",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"]
        },
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "models/conll2003_domain_adapted_bert_base_cased/"
        }
    },
    "encoder": {
      "type": "pass_through",
      "input_dim": 768,
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 16
  },
  "trainer": {
    "optimizer": {
        "type": "bert_adam",
        "lr": 0.00005,
        "warmup": 0.1,
        "t_total": 10000,
        "schedule": "warmup_linear"
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": -1,
    "num_epochs": 4,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
