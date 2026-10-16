from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import RandomSampler, TensorDataset, WeightedRandomSampler

from scripts import evaluate_vit, train_classifier_fixed


def test_candidate_checkpoint_is_separate_from_live_application_model():
    candidate = train_classifier_fixed.OUTPUT_DIR.resolve()
    live = train_classifier_fixed.LIVE_MODEL_DIR.resolve()

    assert candidate != live
    assert not candidate.is_relative_to(live)


def test_evaluator_accepts_explicit_model_directory():
    explicit_model_dir = Path("custom-candidate")

    args = evaluate_vit.build_arg_parser().parse_args(
        ["--model-dir", str(explicit_model_dir)]
    )

    assert args.model_dir == explicit_model_dir
    assert evaluate_vit.build_arg_parser().parse_args([]).model_dir == (
        evaluate_vit.DEFAULT_MODEL_DIR
    )


def test_weighted_cross_entropy_and_ordinary_shuffle_are_configured():
    criterion, weights = train_classifier_fixed.build_weighted_criterion(
        [0, 0, 0, 1],
        torch.device("cpu"),
    )
    loader = train_classifier_fixed.make_shuffled_loader(
        TensorDataset(torch.arange(4)),
        batch_size=2,
        collate_fn=None,
    )

    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.weight is not None
    assert weights == [1.0, 3.0]
    assert isinstance(loader.sampler, RandomSampler)
    assert not isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.sampler.replacement is False


def test_fixed_manifest_and_source_hashes_are_validated_and_recorded(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        train_classifier_fixed,
        "OUTPUT_DIR",
        tmp_path / "vit_candidate",
    )
    train_df, validation_df, provenance = (
        train_classifier_fixed.validate_experiment_inputs()
    )
    args = SimpleNamespace(
        seed=42,
        batch_size=8,
        epochs=5,
        lr=1e-5,
        weight_decay=1e-4,
        freeze_backbone=True,
        max_steps=0,
    )
    run_metadata = train_classifier_fixed.build_run_metadata(
        args,
        provenance,
        "test-run",
    )

    assert len(train_df) == 8110
    assert len(validation_df) == 1728
    assert run_metadata["manifest_sha256"] == {
        "train": train_classifier_fixed.sha256_file(
            train_classifier_fixed.TRAIN_CSV
        ),
        "validation": train_classifier_fixed.sha256_file(
            train_classifier_fixed.VAL_CSV
        ),
        "test": train_classifier_fixed.sha256_file(
            train_classifier_fixed.TEST_CSV
        ),
    }
    assert run_metadata["source_metadata_sha256"] == (
        train_classifier_fixed.sha256_file(
            train_classifier_fixed.BASE_DIR
            / provenance["source_metadata_filename"]
        )
    )
    assert run_metadata["validation_checks_passed"] is True
    assert run_metadata["history"] == []
    assert run_metadata["seed"] == 42
    assert run_metadata["loss"]["name"] == "weighted_cross_entropy"
    assert run_metadata["sampler"]["name"] == "ordinary_shuffle"
    assert run_metadata["sampler"]["weighted_random_sampler"] is False


def test_run_metadata_persistence_writes_json_without_creating_candidate(tmp_path, monkeypatch):
    metadata_path = tmp_path / "run_metadata.json"
    monkeypatch.setattr(
        train_classifier_fixed,
        "RUN_METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        train_classifier_fixed,
        "OUTPUT_DIR",
        tmp_path / "candidate",
    )

    train_classifier_fixed.save_run_metadata(
        {"experiment_id": "test-run", "history": []}
    )

    assert metadata_path.is_file()
    assert not (tmp_path / "candidate").exists()
    assert '"experiment_id": "test-run"' in metadata_path.read_text()


def test_validation_pr_auc_is_the_only_checkpoint_selection_metric():
    assert train_classifier_fixed.MODEL_SELECTION_METRIC == "validation_pr_auc"
    assert train_classifier_fixed.is_better_checkpoint(0.7, 0.6)
    assert not train_classifier_fixed.is_better_checkpoint(0.6, 0.7)

    with pytest.raises(ValueError, match="both binary classes"):
        train_classifier_fixed.compute_validation_metrics(
            labels=[0, 0],
            predictions=[0, 1],
            melanoma_probabilities=[0.1, 0.8],
        )

    with pytest.raises(ValueError, match="finite"):
        train_classifier_fixed.is_better_checkpoint(float("nan"), 0.6)


def test_test_manifest_is_not_read_for_training_or_checkpoint_selection():
    training_tree = ast.parse(inspect.getsource(train_classifier_fixed.train))
    training_names = {
        node.id
        for node in ast.walk(training_tree)
        if isinstance(node, ast.Name)
    }
    selection_tree = ast.parse(
        inspect.getsource(train_classifier_fixed.is_better_checkpoint)
    )
    selection_names = {
        node.id
        for node in ast.walk(selection_tree)
        if isinstance(node, ast.Name)
    }

    assert "TEST_CSV" not in training_names
    assert "test_df" not in training_names
    assert "TEST_CSV" not in selection_names
    assert "test_df" not in selection_names


def test_training_defaults_match_controlled_experiment_configuration():
    args = train_classifier_fixed.build_arg_parser().parse_args([])

    assert args.seed == 42
    assert args.batch_size == 8
    assert args.epochs == 5
    assert args.lr == 1e-5
    assert args.weight_decay == 1e-4
    assert args.freeze_backbone is True
