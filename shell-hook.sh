export PATH="/workspace/.pixi/envs/default/bin:/opt/pixi/bin:/opt/pixi/envs/test/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CONDA_SHLVL=1
export CONDA_PREFIX=/workspace/.pixi/envs/default
export PIXI_IN_SHELL=1
export PIXI_PROJECT_VERSION=0.1.0
export PIXI_PROJECT_NAME=cartpole
export PIXI_EXE=/opt/pixi/bin/pixi
export PIXI_PROJECT_ROOT=/workspace
export PIXI_PROJECT_MANIFEST=/workspace/pixi.toml
export CONDA_DEFAULT_ENV=cartpole
export PIXI_ENVIRONMENT_NAME=default
export PIXI_ENVIRONMENT_PLATFORMS=linux-64
export PIXI_PROMPT='(cartpole) '
. /workspace/.pixi/envs/default/etc/conda/activate.d/libblas_mkl_activate.sh
. /workspace/.pixi/envs/default/etc/conda/activate.d/libglib_activate.sh
. /workspace/.pixi/envs/default/etc/conda/activate.d/libxml2-split_activate.sh
. /workspace/scripts/post-install.bash
source /workspace/.pixi/envs/default/share/bash-completion/completions/*

# shellcheck shell=bash
pixi() {
    local first_arg="${1-}"

    "${PIXI_EXE-}" "$@" || return $?

    case "${first_arg-}" in
    add | a | remove | rm | install | i)
        eval "$("$PIXI_EXE" shell-hook --change-ps1 false)"
        hash -r
        ;;
    esac || :

    return 0
}

export PS1="(cartpole) ${PS1:-}"
exec "$@"
