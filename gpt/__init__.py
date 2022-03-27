CONFIGS = dict(
    batch_size=64,
    seq_length=127,  # 128 - 1
    embedding_dim=64,
    num_embeddings=65,  # vocab size of the data not a hyper_parameter
    n_transformer_layers=8,
    num_attention_heads=16,
    num_weights=64,
    num_epochs=250,
    learning_rate=3e-4,
    shuffle_train_data=True,
    train_log_interval=10,
    seed=5,
)
