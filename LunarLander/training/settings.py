"""Training settings shared by the LunarLander experiment entry points."""

# Absolute horizon within each training stage, independent of evaluation blocks.
EXPLORATION_DECAY_STEPS = 500_000

# Refinement retains exploration without returning to entirely random actions.
ADDITIONAL_EXPLORATION_INITIAL_EPS = 0.4


def selected_training_config(model_sizes, seeds, argv=None):
    """Restrict a training entry point without changing evaluation defaults."""
    import argparse

    parser = argparse.ArgumentParser(description="Train selected LunarLander model sizes.")
    parser.add_argument("--sizes", nargs="+", type=int, choices=model_sizes,
                        default=model_sizes, help="Hidden-layer widths; default: all sizes")
    parser.add_argument("--seeds", nargs="+", type=int, choices=seeds,
                        default=seeds, help="Training seeds; default: all five seeds")
    args = parser.parse_args(argv)
    args.sizes = sorted(set(args.sizes))
    args.seeds = sorted(set(args.seeds))
    return args


def selected_training_sizes(model_sizes, argv=None):
    """Compatibility helper for callers selecting only widths."""
    return selected_training_config(model_sizes, list(range(5)), argv).sizes
