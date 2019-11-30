with import <nixpkgs> {};

let common = [
    gcc9
    gdb
]; 
in stdenvNoCC.mkDerivation rec {
		name = "hm-env";
		buildInputs = common;
}



