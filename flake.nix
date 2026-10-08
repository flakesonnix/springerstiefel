{
  description = "springerstiefel – lokaler OpenAI-kompatibler Gateway für Hey_ (BILD)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Note: the browser revision must match the Playwright version in
      # pyproject.toml. After `nix flake update`, align it there if needed.
      # Current: playwright-driver 1.63.0 <-> playwright>=1.63,<1.64.
      pipNativeLibs = with pkgs; [
        stdenv.cc.cc.lib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          uv
          ruff
          playwright-driver
        ];

        shellHook = ''
          # Native pip extensions (e.g. greenlet) need libstdc++.
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath pipNativeLibs}:$LD_LIBRARY_PATH"

          # Browsers come patched from nixpkgs instead of downloads
          # (`playwright install` doesn't work on NixOS: dynamically linked
          # node + unpatched browser binaries).
          export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"

          # Playwright's bundled node binary doesn't run on NixOS.
          # Replace with the nixpkgs node (it's just a node interpreter).
          if [ -d .venv ]; then
            for target in .venv/lib/python*/site-packages/playwright/driver/node; do
              if [ -f "$target" ] && [ ! -L "$target" ]; then
                ln -sf "${pkgs.nodejs}/bin/node" "$target"
                echo "springerstiefel: playwright-driver/node -> nixpkgs node"
              fi
            done
          fi

          echo "springerstiefel shell – weiter mit:"
          echo "  uv venv .venv && source .venv/bin/activate   # if not done yet"
          echo "  uv pip install -e .   # or: pip install -e ."
        '';
      };
    };
}
