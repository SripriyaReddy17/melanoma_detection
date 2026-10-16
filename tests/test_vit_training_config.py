from scripts import train_classifier


def test_vit_training_targets_40_percent_accuracy():
    assert train_classifier.TARGET_ACCURACY >= 0.40
    assert train_classifier.DEFAULT_EPOCHS >= 5
    assert train_classifier.DEFAULT_LR <= 2e-5


def test_freeze_backbone_option_keeps_head_trainable():
    model = train_classifier.build_model()
    train_classifier.set_trainable_parameters(model, freeze_backbone=True)
    assert model.classifier.weight.requires_grad is True
    assert any(param.requires_grad is False for param in model.vit.parameters())
