#!/usr/bin/env bash
# Source in the active training Python environment; settings live in ../.env.
if [[ -n "${BASH_VERSION:-}" ]]; then
  _wbwam_loader_file="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _wbwam_loader_file="${(%):-%x}"
else
  echo "load_env.sh requires Bash or Zsh" >&2
  return 1
fi
if ! _wbwam_exports="$(python "$(dirname -- "${_wbwam_loader_file}")/load_env.py")"; then
  unset _wbwam_loader_file _wbwam_exports
  return 1
fi
eval "${_wbwam_exports}"
unset _wbwam_loader_file _wbwam_exports
