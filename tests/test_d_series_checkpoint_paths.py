from cmgm.scripts.main_ablation import _checkpoint_path_for_variant


def test_only_d_series_variants_receive_deterministic_checkpoint_paths(tmp_path):
    path = _checkpoint_path_for_variant(
        "switching_latent_balanced_readout", tmp_path / "nested"
    )
    assert path == tmp_path / "nested" / "switching_latent_balanced_readout_best.pt"
    assert path.parent.is_dir()
    d0d = _checkpoint_path_for_variant(
        "switching_latent_balanced_transition", tmp_path
    )
    assert d0d.name == "switching_latent_balanced_transition_best.pt"
    assert _checkpoint_path_for_variant("market_token_transformer", tmp_path) is None
