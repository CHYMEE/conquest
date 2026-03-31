import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "run_quick_tune_qpso_lstm_sgd.py",
        "--task",
        "binary",
        "--feature-method",
        "variance",
        "--subsample-size",
        "80000",
        "--feature-choices",
        "40,60,80",
        "--n-trials",
        "8",
        "--epochs",
        "35",
        "--patience",
        "6",
        "--output-dir",
        "outputs/quick_tune_binary_variance_hi",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
