#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/release.sh <version> [--commit] [--tag] [--push]

Examples:
  scripts/release.sh 0.1.3
  scripts/release.sh 0.1.3 --commit --tag
  scripts/release.sh 0.1.3 --commit --tag --push

Behavior:
  - Updates:
      - pyproject.toml -> [tool.poetry].version
      - src/fraclab_sdk/version.py -> __version__
  - Optional:
      --commit : create commit "release: v<version>"
      --tag    : create annotated tag "v<version>" (requires --commit)
      --push   : push main + tags (requires --tag)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

VERSION="$1"
shift

DO_COMMIT=0
DO_TAG=0
DO_PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) DO_COMMIT=1 ;;
    --tag) DO_TAG=1 ;;
    --push) DO_PUSH=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ $DO_TAG -eq 1 && $DO_COMMIT -ne 1 ]]; then
  echo "--tag requires --commit" >&2
  exit 1
fi

if [[ $DO_PUSH -eq 1 && $DO_TAG -ne 1 ]]; then
  echo "--push requires --tag" >&2
  exit 1
fi

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "Invalid version: $VERSION" >&2
  echo "Expected semver-like format, e.g. 0.1.3 or 1.2.3-rc.1" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]]; then
  echo "pyproject.toml not found in repo root: $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -f src/fraclab_sdk/version.py ]]; then
  echo "src/fraclab_sdk/version.py not found" >&2
  exit 1
fi

if [[ $DO_COMMIT -eq 1 ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree is not clean. Commit/stash changes before release bump." >&2
    exit 1
  fi
fi

# Update [tool.poetry].version in pyproject.toml
awk -v new_version="$VERSION" '
  BEGIN { in_poetry=0; changed=0 }
  /^\[tool\.poetry\]/ { in_poetry=1; print; next }
  /^\[/ && $0 !~ /^\[tool\.poetry\]/ { in_poetry=0 }
  in_poetry && $0 ~ /^version = "/ {
    print "version = \"" new_version "\""
    changed=1
    next
  }
  { print }
  END {
    if (changed==0) {
      print "Failed to locate [tool.poetry] version in pyproject.toml" > "/dev/stderr"
      exit 2
    }
  }
' pyproject.toml > pyproject.toml.tmp
mv pyproject.toml.tmp pyproject.toml

# Update __version__ in module metadata
awk -v new_version="$VERSION" '
  BEGIN { changed=0 }
  /^__version__ = "/ {
    print "__version__ = \"" new_version "\""
    changed=1
    next
  }
  { print }
  END {
    if (changed==0) {
      print "Failed to locate __version__ in src/fraclab_sdk/version.py" > "/dev/stderr"
      exit 2
    }
  }
' src/fraclab_sdk/version.py > src/fraclab_sdk/version.py.tmp
mv src/fraclab_sdk/version.py.tmp src/fraclab_sdk/version.py

echo "Updated version to $VERSION:"
echo "  - pyproject.toml"
echo "  - src/fraclab_sdk/version.py"

if [[ $DO_COMMIT -eq 1 ]]; then
  git add pyproject.toml src/fraclab_sdk/version.py
  git commit -m "release: v$VERSION"
  echo "Created commit: release: v$VERSION"
fi

if [[ $DO_TAG -eq 1 ]]; then
  git tag -a "v$VERSION" -m "Release v$VERSION"
  echo "Created tag: v$VERSION"
fi

if [[ $DO_PUSH -eq 1 ]]; then
  git push origin main --follow-tags
  echo "Pushed main and tags to origin"
fi
