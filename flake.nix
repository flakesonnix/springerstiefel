{
  description = "springerstiefel – lokaler OpenAI-kompatibler Gateway für Hey_ (BILD)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Hinweis: Die Browser-Revision muss zur Playwright-Version in
      # pyproject.toml passen. Nach `nix flake update` ggf. dort angleichen.
      # Aktuell: playwright-driver 1.63.0 <-> playwright>=1.63,<1.64.
      pipNativeLibs = with pkgs; [
        stdenv.cc.cc.lib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          uv
          playwright-driver
        ];

        shellHook = ''
          # Native pip-Extensions (z.B. greenlet) brauchen libstdc++.
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath pipNativeLibs}:$LD_LIBRARY_PATH"

          # Browser kommen gepatcht aus nixpkgs statt per Download
          # (`playwright install` läuft auf NixOS nicht: dynamisch gelinktes
          # node + unpatchte Browser-Binaries).
          export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"

          # Playwrights gebündeltes node-Binary läuft auf NixOS nicht.
          # Durch nixpkgs-node ersetzen (ist nur ein node-Interpreter).
          if [ -d .venv ]; then
            for target in .venv/lib/python*/site-packages/playwright/driver/node; do
              if [ -f "$target" ] && [ ! -L "$target" ]; then
                ln -sf "${pkgs.nodejs}/bin/node" "$target"
                echo "springerstiefel: playwright-driver/node -> nixpkgs node"
              fi
            done
          fi

          echo "springerstiefel shell – weiter mit:"
          echo "  uv venv .venv && source .venv/bin/activate   # falls noch nicht geschehen"
          echo "  uv pip install -e .   # oder: pip install -e ."
        '';
      };
    };
}
