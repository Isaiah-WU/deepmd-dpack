#!/usr/bin/env bash
# dpack bootstrap installer — like dp1s / pixi / rustup.
#
#   curl -fsSL https://raw.githubusercontent.com/Isaiah-WU/deepmd-dpack/main/install.sh | bash
#
# Installs the `dpack` command into your USER directory (no root needed),
# then you run:  dpack install dp
set -euo pipefail

REPO="Isaiah-WU/deepmd-dpack"
DPACK_URL="https://raw.githubusercontent.com/${REPO}/main/dpack"

# Install to user bin (no root). Honor XDG, fall back to ~/.local/bin.
BIN_DIR="${DPACK_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$BIN_DIR"

echo "==> Downloading dpack to $BIN_DIR/dpack"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$DPACK_URL" -o "$BIN_DIR/dpack"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "$BIN_DIR/dpack" "$DPACK_URL"
else
  echo "ERROR: need curl or wget" >&2; exit 1
fi
chmod +x "$BIN_DIR/dpack"

# Make sure it's on PATH for this and future shells.
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;  # already on PATH
  *)
    echo "==> Adding $BIN_DIR to PATH in your shell rc"
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
      [[ -f "$rc" ]] || continue
      grep -q "$BIN_DIR" "$rc" 2>/dev/null || \
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$rc"
    done
    export PATH="$BIN_DIR:$PATH"
    ;;
esac

echo ""
echo "✓ dpack installed to $BIN_DIR/dpack"
echo ""
echo "Next:"
echo "  dpack install dp                 # online: auto-detect GPU, download + install"
echo "  dpack install dp --file ./x.sh   # offline: install from a local .sh"
echo ""
echo "If 'dpack' is not found, run:  export PATH=\"$BIN_DIR:\$PATH\"  (or open a new shell)"
