from LunarLander.paths import APP_ROOT
import os
from pathlib import Path


MODEL_DIRS = {
    "base_float": Path(f"{APP_ROOT}/models/initial_fp32"),
    "qat_weight_only": Path(f"{APP_ROOT}/models/qat_w8"),
    "qat_weight_activation": Path(f"{APP_ROOT}/models/qat_w8a8"),
}


def file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main():
    for model_type, folder in MODEL_DIRS.items():
        print("=" * 70)
        print(f"{model_type}: {folder}")
        print("=" * 70)

        zip_files = sorted(folder.glob("*.zip"))
        pt_files = sorted(folder.glob("*_int8_weights.pt"))

        if zip_files:
            print("\nFloat checkpoints (.zip)")
            for path in zip_files:
                size_mb = file_size_mb(path)
                print(f"{path.name:<45} {size_mb:>8.3f} MB")

        if pt_files:
            print("\nInt8 weights (.pt)")
            for path in pt_files:
                size_mb = file_size_mb(path)
                print(f"{path.name:<45} {size_mb:>8.3f} MB")

        if model_type != "base_float":
            print("\nFloat vs Int8 comparison")
            for zip_path in zip_files:
                int8_path = folder / f"{zip_path.stem}_int8_weights.pt"

                if not int8_path.exists():
                    print(f"{zip_path.name:<45} -> no matching int8 file")
                    continue

                float_size = file_size_mb(zip_path)
                int8_size = file_size_mb(int8_path)
                reduction = (1 - int8_size / float_size) * 100

                print(f"{zip_path.stem}")
                print(f"  float: {float_size:.3f} MB")
                print(f"  int8:  {int8_size:.3f} MB")
                print(f"  saved: {reduction:.1f}%")

        print()


if __name__ == "__main__":
    main()