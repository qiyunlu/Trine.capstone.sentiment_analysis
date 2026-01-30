{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    git
    curl
  ];

  shellHook = ''
    echo "Nix devShell Activated"
  '';
}
