import os
import subprocess
import sys


def run_cmd(cmd, capture_output=False, env=None):
    # Run shell commands
    return subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_not_exist = run_cmd("conda", capture_output=True).returncode
    if conda_not_exist:
        print("Conda is not installed. Exiting...")
        sys.exit()

    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit()


def install_dependencies():
    run_cmd("pip install -r requirements.txt")


def run_model():
    run_cmd("python webui.py")


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    # Install all dependencies
    install_dependencies()

    # Run the model with webui
    run_model()
