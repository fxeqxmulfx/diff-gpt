{
  task: "arithmetic-sum-pow4",
  model: "complex-peri-rms-ordinary-v1",
  adam_betas: [0.9,0.95], weight_decay: 0.1,
  warmup_updates: 2000, decay_updates: 20000, decay_shape: "cosine",
  peak_lr: $peak, budget_updates: 100000,
  source_sha256, completed, integrity_failures,
  results: [.results[] | {
    seed, status,
    first99_epochs: .first99.epochs,
    first99_update: .first99.step,
    first99_accuracy: .first99.validation.accuracy,
    previous_observed_epochs,
    best_accuracy: .best.validation.accuracy,
    best_epoch: .best.epochs,
    final_accuracy: .endpoint.validation.accuracy,
    final_epochs: .endpoint.epochs,
    completed_updates: .endpoint.step,
    seconds: .endpoint.seconds
  }]
}
