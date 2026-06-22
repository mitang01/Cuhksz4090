#!/bin/bash
#SBATCH --job-name=install_data_proc_deps
#SBATCH --output=/share/home/mitan/spike_sorting/log_install_data_proc_%j.out
#SBATCH --error=/share/home/mitan/spike_sorting/log_install_data_proc_%j.err
#SBATCH --partition=gpu3node
#SBATCH --nodelist=gpu05
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="/share/home/mitan/spike_sorting"
ENV_FILE="${PROJECT_DIR}/environment.yml"
TARGET_ENV="data-proc"
SANITIZED_ENV_FILE="${PROJECT_DIR}/environment.data-proc.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: Cannot find ${ENV_FILE}"
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  :
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/opt/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found in PATH and no conda.sh found."
  exit 1
fi

echo "Using env file: ${ENV_FILE}"
echo "Target env: ${TARGET_ENV}"

# Remove hardcoded "name"/"prefix" and macOS-only build pins from source YAML.
# This makes the environment solvable on linux-64 clusters.
python - <<'PY'
from pathlib import Path

source = Path("/share/home/mitan/spike_sorting/environment.yml")
target = Path("/share/home/mitan/spike_sorting/environment.data-proc.yml")

lines = source.read_text().splitlines()
out = ["name: data-proc"]
in_deps = False
in_pip = False

# macOS-specific packages that should not be forced on Linux
drop_conda_pkgs = {"libcxx"}
drop_pip_pkgs = {"appnope"}
rewrite_pip_specs = {
    # This release was removed from common indexes; pick closest available pre-release.
    "phy==2.0b6": "phy==2.1.0rc1",
    # Let pip resolve a compatible PyOpenGL version with phy/phylib stack.
    "pyopengl==3.1.6": "pyopengl",
}

def sanitize_conda_dep(dep_line: str):
    raw = dep_line.strip()
    if not raw.startswith("- "):
        return dep_line
    spec = raw[2:].strip()
    if not spec or spec.startswith(("pip:", "{", "[")):
        return dep_line
    if any(ch in spec for ch in "<>!~ "):
        return dep_line

    # Convert "pkg=ver=build" -> "pkg=ver" for cross-platform compatibility.
    parts = spec.split("=")
    pkg = parts[0].strip()
    if pkg in drop_conda_pkgs:
        return None
    if len(parts) >= 3:
        spec = "=".join(parts[:2]).strip()
    return f"  - {spec}"

def sanitize_pip_dep(dep_line: str):
    raw = dep_line.strip()
    if not raw.startswith("- "):
        return dep_line
    spec = raw[2:].strip()
    if not spec:
        return dep_line
    name = spec.split("==", 1)[0].strip().lower()
    if name in drop_pip_pkgs:
        return None
    if spec in rewrite_pip_specs:
        return f"      - {rewrite_pip_specs[spec]}"
    return dep_line

for line in lines:
    stripped = line.strip()
    if stripped.startswith("name:"):
        continue
    if stripped.startswith("prefix:"):
        continue

    if stripped.startswith("dependencies:"):
        in_deps = True
        in_pip = False
        out.append(line)
        continue

    if in_deps and stripped.startswith("- pip:"):
        in_pip = True
        out.append(line)
        continue

    # Leaving dependencies section (back to top-level key)
    if in_deps and line and not line.startswith(" "):
        in_deps = False
        in_pip = False

    if in_deps and not in_pip and line.startswith("  - "):
        cleaned = sanitize_conda_dep(line)
        if cleaned is not None:
            out.append(cleaned)
        continue

    if in_deps and in_pip and line.startswith("      - "):
        cleaned = sanitize_pip_dep(line)
        if cleaned is not None:
            out.append(cleaned)
        continue

    out.append(line)

target.write_text("\n".join(out) + "\n")
print(f"Sanitized environment file written to: {target}")
PY

# Create env if missing, otherwise update existing one.
if conda env list | awk '{print $1}' | rg -x "${TARGET_ENV}" >/dev/null 2>&1; then
  echo "Environment '${TARGET_ENV}' exists. Updating..."
  conda env update --name "${TARGET_ENV}" --file "${SANITIZED_ENV_FILE}" --prune
else
  echo "Environment '${TARGET_ENV}' does not exist. Creating..."
  conda env create --name "${TARGET_ENV}" --file "${SANITIZED_ENV_FILE}"
fi

echo "Dependency installation completed successfully."
