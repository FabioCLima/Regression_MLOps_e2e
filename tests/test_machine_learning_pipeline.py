from pipelines.machine_learning_pipeline import run_machine_learning_pipeline


def test_run_machine_learning_pipeline_calls_enabled_steps(monkeypatch, tmp_path):
    called_steps: list[str] = []

    monkeypatch.setattr(
        "pipelines.machine_learning_pipeline.setup_logger",
        lambda *args, **kwargs: called_steps.append("setup_logger"),
    )
    monkeypatch.setattr(
        "pipelines.machine_learning_pipeline.run_data_pipeline",
        lambda configure_logging: called_steps.append("data"),
    )
    monkeypatch.setattr(
        "pipelines.machine_learning_pipeline.run_feature_pipeline",
        lambda configure_logging: called_steps.append("features"),
    )
    monkeypatch.setattr(
        "pipelines.machine_learning_pipeline.run_training_pipeline",
        lambda configure_logging: called_steps.append("training"),
    )
    monkeypatch.setattr(
        "pipelines.machine_learning_pipeline.run_tuning_pipeline",
        lambda configure_logging: called_steps.append("tuning"),
    )
    monkeypatch.setattr(
        "pipelines.machine_learning_pipeline.run_inference_pipeline",
        lambda **kwargs: called_steps.append("inference"),
    )

    run_machine_learning_pipeline(
        run_data=True,
        run_features=True,
        run_training=False,
        run_tuning=True,
        run_inference=True,
        inference_input_path=tmp_path / "input.csv",
        inference_output_dir=tmp_path,
    )

    assert called_steps == [
        "setup_logger",
        "data",
        "features",
        "tuning",
        "inference",
    ]
