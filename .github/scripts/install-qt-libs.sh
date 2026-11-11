#!/usr/bin/env bash
# Install the system libraries PySide6 needs on Ubuntu runners.
#
# The .debs apt downloads are kept in ~/qt-debs, which the workflow caches
# with actions/cache, so a later run installs them offline when runner DNS to
# the Ubuntu mirrors is down. A cached set that no longer installs cleanly
# falls back to apt.
set -uo pipefail

PKGS=(libegl1 libgl1 libxkbcommon0 libdbus-1-3)
DEB_DIR="$HOME/qt-debs" # the path the workflows cache

if compgen -G "$DEB_DIR/*.deb" >/dev/null; then
  if sudo dpkg -i "$DEB_DIR"/*.deb && dpkg -s "${PKGS[@]}" >/dev/null; then
    echo "Installed Qt system libraries from cached debs"
    exit 0
  fi
  echo "Cached debs did not install cleanly; falling back to apt"
fi

mkdir -p "$DEB_DIR/partial"
for i in 1 2 3 4 5; do
  if sudo apt-get -o Acquire::Retries=3 update &&
    sudo apt-get -o Acquire::Retries=3 -o Dir::Cache::Archives="$DEB_DIR" \
      install -y "${PKGS[@]}"; then
    sudo rm -rf "$DEB_DIR/partial" "$DEB_DIR/lock"
    sudo chown -R "$(id -u):$(id -g)" "$DEB_DIR"
    ls "$DEB_DIR"
    exit 0
  fi
  echo "apt attempt $i failed; retrying in $((i * 30)) s"
  sleep $((i * 30))
done
exit 1
