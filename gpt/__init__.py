CONFIGS = dict(
    batch_size=64,
    seq_length=127,  # 128 - 1
    embedding_dim=16,
    num_embeddings=65,  # vocab size of the data not a hyper_parameter
    n_transformer_layers=3,
    num_attention_heads=8,
    num_attention_features=32,
    num_epochs=100,
    learning_rate=3e-4,
    shuffle_train_data=True,
)
