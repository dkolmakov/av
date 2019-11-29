with import <nixpkgs> {};

let common = [
    gcc8
    gdb
]; 
in stdenvNoCC.mkDerivation rec {
		name = "hm-env";
		buildInputs = common;
}



