{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.0.tar.gz"; # unstable Nixpkgs
  inputs = {
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self
    , nixpkgs
    , flake-parts
    , pyproject-nix
    , uv2nix
    , pyproject-build-systems
    , ...
    }@inputs:

    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      lib = inputs.nixpkgs.lib;
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                #cudaSupport = true;
              };
            };
          }
        );

      version = "3.13";
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        let
          concatMajorMinor =
            v:
            pkgs.lib.pipe v [
              pkgs.lib.versions.splitVersion
              (pkgs.lib.sublist 0 2)
              pkgs.lib.concatStrings
            ];

          python = pkgs."python${concatMajorMinor version}";

          workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = ./.;
          };

          overlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          editableOverlay = workspace.mkEditablePyprojectOverlay {
            root = "$REPO_ROOT";
          };

          customOverlay = self: super: {
            torch = super.torch.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
                pkgs.cudaPackages.libcusparse
                pkgs.cudaPackages.libcusparse_lt
                pkgs.cudaPackages.libcufile
                pkgs.cudaPackages.libnvshmem
                pkgs.cudaPackages.nccl
              ];
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [ "libcuda.so.1" ];
            });
            nvidia-cufile-cu12 = super.nvidia-cufile-cu12.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.rdma-core ];
            });
            nvidia-nvshmem-cu12 = super.nvidia-nvshmem-cu12.overrideAttrs (old: {
              nativeBuildInputs = old.nativeBuildInputs ++ [
                pkgs.openmpi
                pkgs.pmix
                pkgs.ucx
                pkgs.libfabric
                pkgs.rdma-core
              ];
            });
            nvidia-cusparse-cu12 = super.nvidia-cusparse-cu12.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.cudaPackages.libnvjitlink ];
            });
            nvidia-cusolver-cu12 = super.nvidia-cusolver-cu12.overrideAttrs (old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                pkgs.cudaPackages.libnvjitlink
                pkgs.cudaPackages.libcusparse
                pkgs.cudaPackages.libcublas
              ];
            });
          };

          pythonSets =
            (pkgs.callPackage pyproject-nix.build.packages {
              inherit python;
            }).overrideScope
              (
                pkgs.lib.composeManyExtensions [
                  pyproject-build-systems.overlays.wheel
                  overlay
                  customOverlay
                ]
              );
        in
        {
          default = let
            pythonSet = pythonSets.overrideScope editableOverlay;
            virtualenv = pythonSet.mkVirtualEnv "my-python-env" workspace.deps.default;
          in pkgs.mkShell {
            venvDir = ".venv";

            # Ensure libstdc++, zlib, and the system NVIDIA driver (libcuda.so) are available
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:/run/opengl-driver/lib";

            packages = [
              virtualenv
              pkgs.uv
              pkgs.cacert
              pkgs.ffmpeg
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.libcusolver
              pkgs.cudaPackages.cuda_cudart
            ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              REQUESTS_CA_BUNDLE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            };

            shellHook = ''
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              export LD_LIBRARY_PATH="${lib.makeLibraryPath [
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
                pkgs.cudaPackages.cuda_nvcc
                pkgs.cudaPackages.libcublas
                pkgs.cudaPackages.libcusolver
                pkgs.cudaPackages.cuda_cudart
              ]}:$LD_LIBRARY_PATH"
              echo "UV_PYTHON -> $UV_PYTHON"
              echo "LD_LIBRARY_PATH -> $LD_LIBRARY_PATH"
              unset PYTHONPATH
            '';
          };
        }
      );
    };
}
