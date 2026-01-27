#!/usr/bin/env bash
set -euo pipefail

EXPERIMENTS_DIR="./experiments"
SCRATCH_ROOT="/scratch/ml8736/memory-replay"
LINK_NAMES=(logs figures saved_models)

if [[ ! -d "${EXPERIMENTS_DIR}" ]]; then
  echo "ERROR: '${EXPERIMENTS_DIR}' does not exist (run from repo root?)" >&2
  exit 1
fi

for exp_dir in "${EXPERIMENTS_DIR}"/*; do
  [[ -d "${exp_dir}" ]] || continue
  exp_name="$(basename "${exp_dir}")"

  for link_name in "${LINK_NAMES[@]}"; do
    src="${SCRATCH_ROOT}/${exp_name}/${link_name}"
    dst="${exp_dir}/${link_name}"

    # If already correct, do nothing.
    if [[ -L "${dst}" ]] && [[ "$(readlink "${dst}")" == "${src}" ]]; then
      continue
    fi

    # Safety: never delete real files/dirs; only replace symlinks.
    if [[ -e "${dst}" && ! -L "${dst}" ]]; then
      echo "WARN: '${dst}' exists and is not a symlink; skipping." >&2
      continue
    fi

    # Ensure scratch target exists.
    mkdir -p "${src}"

    # Replace existing symlink (if any) and create the new one.
    rm -f "${dst}"
    ln -s "${src}" "${dst}"
    echo "linked: ${dst} -> ${src}"
  done
done
