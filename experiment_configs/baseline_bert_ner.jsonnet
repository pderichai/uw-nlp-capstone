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
         "pretrained_model": "bert-base-cased"
     },
    }
  },
  "train_data_path": std.extVar("NER_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("NER_TEST_A_PATH"),
  "model": {
    "type": "simple_tagger",
    "label_encoding": "IOB1",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"]
        },
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-large-cased"

        }
    }
    "encoder": {
      "type": "pass_through",
      "input_size": 768,
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 80
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}