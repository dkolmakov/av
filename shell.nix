with import <nixpkgs> {};

{

eight = stdenvNoCC.mkDerivation rec {
		name = "hm-env";
		buildInputs = [gcc9 gdb];
};

nine = stdenvNoCC.mkDerivation rec {
		name = "hm-env";
		buildInputs = [gcc9 gdb];
};

}

