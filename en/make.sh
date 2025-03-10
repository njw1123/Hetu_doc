#!/bin/bash

# 切换到脚本所在目录
cd "$(dirname "$0")" || exit 1

# 配置 Sphinx 路径（支持自定义环境变量）
SPHINXBUILD="${SPHINXBUILD:-sphinx-build}"
SOURCEDIR="source"
BUILDDIR="build"

# 检查是否传入参数，若无则显示帮助
if [ $# -eq 0 ]; then
    echo "Error: Missing build target (e.g. 'html', 'clean')"
    echo "Use '$0 help' for available targets"
    exit 1
fi

# 检查 sphinx-build 是否存在
if ! command -v "$SPHINXBUILD" >/dev/null 2>&1; then
    echo ""
    echo "Error: The 'sphinx-build' command was not found."
    echo "Either:"
    echo "1. Install Sphinx via pip: 'pip install sphinx'"
    echo "2. Set SPHINXBUILD env variable to the full path of sphinx-build"
    echo "   Example (if using virtualenv):"
    echo '   export SPHINXBUILD="$VIRTUAL_ENV/bin/sphinx-build"'
    echo ""
    exit 1
fi

# 执行构建命令（自动传递所有参数）
"$SPHINXBUILD" -M "$1" "$SOURCEDIR" "$BUILDDIR" ${SPHINXOPTS} ${O}

# 提示：如果遇到权限问题，运行 chmod +x make.sh