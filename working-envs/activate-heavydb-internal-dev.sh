#
# Prepare heavyaidb development environment, detect CUDA availability
#
# Usage:
#  source <this file.sh>
#
# Assumptions:
#   Existence of /usr/local/cuda-10.1.243/env.sh
#   Existence of heavyaidb-cuda-dev or heavyaidb-cpu-dev conda environment
#
# Author: Pearu Peterson
# Created: November 2019
#

CORES_PER_SOCKET=`lscpu | grep 'Core(s) per socket' | awk '{print $NF}'`
NUMBER_OF_SOCKETS=`lscpu | grep 'Socket(s)' | awk '{print $NF}'`
export NCORES=`echo "$CORES_PER_SOCKET * $NUMBER_OF_SOCKETS"| bc`

export CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=release -DMAPD_EDITION=EE -DMAPD_DOCS_DOWNLOAD=off -DENABLE_AWS_S3=off -DENABLE_FOLLY=ON -DENABLE_JAVA_REMOTE_DEBUG=off -DENABLE_PROFILER=off -DPREFER_STATIC_LIBS=off -DENABLE_AWS_S3=OFF -DENABLE_FSI_ODBC=OFF"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_WARNINGS_AS_ERRORS=OFF"
export CMAKE_OPTIONS_CUDA_EXTRA=""
export CMAKE_OPTIONS_NOCUDA_EXTRA="-DENABLE_CUDA=OFF"
export CMAKE_OPTIONS_DBE_EXTRA="-DENABLE_DBE=ON -DENABLE_FSI=ON -DENABLE_ITT=OFF -DENABLE_JIT_DEBUG=OFF -DENABLE_INTEL_JIT_LISTENER=OFF -DENABLE_TESTS=OFF"
export CMAKE_OPTIONS_TSAN_EXTRA="-DENABLE_FOLLY=OFF -DENABLE_TSAN=ON -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=RELWITHDEBINFO"


# Temporarily disable GEOS due to https://github.com/xnd-project/rbc/issues/196
# export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_GEOS=off"

CONDA_ENV_LIST=$(conda env list | awk '{print $1}' )

if [[ -x "$(command -v nvidia-smi)" ]]
then
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/set_cuda_env.sh
    # read set_cuda_env.sh reader

    if [[ -f /usr/local/cuda-11.7.1/env.sh ]]
    then
        CUDA_VERSION=${CUDA_VERSION:-11.7.1}
    elif [[ -f /usr/local/cuda-11.5.0/env.sh ]]
    then
        CUDA_VERSION=${CUDA_VERSION:-11.5.0}
    elif [[ -f /usr/local/cuda-11.0.3/env.sh ]]
    then
        CUDA_VERSION=${CUDA_VERSION:-11.0.3}
    elif [[ -f /usr/local/cuda-10.2.89/env.sh ]]
    then
        CUDA_VERSION=${CUDA_VERSION:-10.2.89}
    else
        CUDA_VERSION=${CUDA_VERSION:-10.1.243}
    fi
    source /usr/local/cuda-${CUDA_VERSION}/env.sh

    export CMAKE_OPTIONS_CUDA_EXTRA="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DENABLE_CUDA=on"
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/heavydb-dev.yaml
    # conda env create  --file=heavydb-dev.yaml -n heavydb-cuda-dev
    #
    # conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/heavydb-dev.yaml -n heavydb-cuda-dev
    #
    # conda install -y -n heavydb-cuda-dev -c conda-forge nvcc_linux-64
    USE_ENV="${USE_ENV:-heavydb-cuda-dev}"

    if [[ $CONDA_ENV_LIST = *"$USE_ENV"* ]]
    then
        if [[ "$CONDA_DEFAULT_ENV" = "$USE_ENV" ]]
        then
            echo "deactivating $USE_ENV"
            conda deactivate
        fi
        if [[ -n "$(type -t layout_conda)" ]]
        then
            layout_conda $USE_ENV
        else
            conda activate $USE_ENV
        fi
    else
        echo "conda environment does not exist. To create $USE_ENV, run:"
        echo "mamba env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/heavydb-dev.yaml -n $USE_ENV"
        exit 1
    fi
    export CXXFLAGS="$CXXFLAGS -I$CUDA_HOME/include"
    export CPPFLAGS="$CPPFLAGS -I$CUDA_HOME/include"
    export CFLAGS="$CFLAGS -I$CUDA_HOME/include"
    export LDFLAGS="${LDFLAGS} -Wl,-rpath,${CUDA_HOME}/lib64 -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64"

else
    # wget https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/conda-envs/heavydb-dev.yaml
    # conda env create  --file=heavydb-dev.yaml -n heavydb-cpu-dev
    USE_ENV="${USE_ENV:-heavydb-cpu-dev}"

    if [[ $CONDA_ENV_LIST = *"$USE_ENV"* ]]
    then
        if [[ "$CONDA_DEFAULT_ENV" = "$USE_ENV" ]]
        then
            echo "deactivating $USE_ENV"
            conda deactivate
        fi
        if [[ -n "$(type -t layout_conda)" ]]; then
            layout_conda $USE_ENV
        else
            conda activate $USE_ENV
        fi
    else
        echo "conda environment does not exist. To create $USE_ENV, run:"
        echo "conda env create  --file=~/git/Quansight/pearu-sandbox/conda-envs/heavydb-cpu-dev.yaml -n $USE_ENV"
        exit 1
    fi
fi

# Fixes H3LibExtFuncTest.cpp.o: undefined reference to symbol 'curl_easy_init'
export LDFLAGS="${LDFLAGS} -lcurl"
# Remove --as-needed to resolve undefined reference to `__vdso_clock_gettime@GLIBC_PRIVATE'
export LDFLAGS="`echo $LDFLAGS | sed 's/-Wl,--as-needed//'`"

# Using conda ld (make -j18 -> 12m25s/113m40s; change heavydbTypes.h -> 2m14s/10m18s)

# Use gold linker (make -j18 -> 12m9s)
# export LDFLAGS="${LDFLAGS} -fuse-ld=gold -Wl,--threads -Wl,--thread-count=$NCORES"

# Use mold linker via replacing compiler linker ($LD) with ld.mold
# manually in <conda env>/x86_64-conda-linux-gnu/bin:
#  mv ld ld.orig;
#  ln -s /usr/local/bin/ld.mold ld
# (make -j18 -> 12m5s/107m25; change heavydbTypes.h -> 1m48s/3m46s)
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_LINKER=/usr/local/bin/ld.mold"  # this might not be necessary!

# export CXXFLAGS="$CXXFLAGS -include $HOME/git/Quansight/pearu-sandbox/cxx/toString.hpp"
# export CXXFLAGS="$CXXFLAGS -include $HOME/git/Quansight/pearu-sandbox/cxx/toString_utils.hpp"
export CXXFLAGS="$CXXFLAGS -DENABLE_TOSTRING_LLVM"
export CXXFLAGS="$CXXFLAGS -DENABLE_TOSTRING_RAPIDJSON"
export CXXFLAGS="$CXXFLAGS -DENABLE_TOSTRING_str"
export CXXFLAGS="$CXXFLAGS -DENABLE_TOSTRING_to_string"
# export CXXFLAGS="$CXXFLAGS -DENABLE_TOSTRING_PTHREAD"

# resolve undefined reference to `llvm::DisableABIBreakingChecks'
export CXXFLAGS="$CXXFLAGS -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1"

export CONDA_BUILD_SYSROOT=$CONDA_PREFIX/$HOST/sysroot

export CXXFLAGS="`echo $CXXFLAGS | sed 's/-fPIC//'`"
export CXXFLAGS="$CXXFLAGS -DBOOST_ERROR_CODE_HEADER_ONLY"
export CXXFLAGS="$CXXFLAGS -DBOOST_DISABLE_PRAGMA_MESSAGE"
export CXXFLAGS="$CXXFLAGS -D__STDC_FORMAT_MACROS"
export CXXFLAGS="$CXXFLAGS -Dsecure_getenv=getenv"
export CXXFLAGS="$CXXFLAGS -DFLATBUFFER_ERROR_ABORTS"
export CFLAGS="$CFLAGS -DFLATBUFFER_ERROR_ABORTS"

# export CXXFLAGS="$CXXFLAGS -g"

# export CC=$CONDA_PREFIX/bin/clang
# export CXX=$CONDA_PREFIX/bin/clang++

export CMAKE_CC="${CMAKE_CC:-$CC}"
export CMAKE_CXX="${CMAKE_CXX:-$CXX}"

export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_C_COMPILER=$CMAKE_CC -DCMAKE_CXX_COMPILER=$CMAKE_CXX"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_SYSROOT=$CONDA_BUILD_SYSROOT"
export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_TESTS=on"
#export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_OMNIVERSE_CONNECTOR=off"
#export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_POINT_CLOUD_TFS=off"
#export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_RF_PROP_TFS=off"
# export CMAKE_OPTIONS="$CMAKE_OPTIONS -DENABLE_SYSTEM_TFS=off"

export CMAKE_OPTIONS_NOCUDA="$CMAKE_OPTIONS $CMAKE_OPTIONS_NOCUDA_EXTRA"
export CMAKE_OPTIONS_CUDA="$CMAKE_OPTIONS $CMAKE_OPTIONS_CUDA_EXTRA"
export CMAKE_OPTIONS_CUDA_DBE="$CMAKE_OPTIONS $CMAKE_OPTIONS_CUDA_EXTRA $CMAKE_OPTIONS_DBE_EXTRA"
export CMAKE_OPTIONS_NOCUDA_DBE="$CMAKE_OPTIONS $CMAKE_OPTIONS_NOCUDA_EXTRA $CMAKE_OPTIONS_DBE_EXTRA"
export CMAKE_OPTIONS_NOCUDA_MLPACK="$CMAKE_OPTIONS_NOCUDA -DENABLE_MLPACK=ON"
export CMAKE_OPTIONS_TSAN="$CMAKE_OPTIONS $CMAKE_OPTIONS_TSAN_EXTRA"

# resolves `fatal error: boost/regex.hpp: No such file or directory`
echo -e "#!/bin/sh\n${CUDA_HOME}/bin/nvcc -ccbin $CC -v \$@" > $PWD/nvcc
chmod +x $PWD/nvcc
export PATH=$PWD:$PATH

# resolves UdfTest fatal error: 'cstdint' file not found
test -f nvcc-boost-include-dirs.patch || wget https://raw.githubusercontent.com/conda-forge/omniscidb-cuda-feedstock/master/recipe/recipe/nvcc-boost-include-dirs.patch
test -f get_cxx_include_path.sh || wget https://raw.githubusercontent.com/conda-forge/omniscidb-cuda-feedstock/master/recipe/recipe/get_cxx_include_path.sh
. get_cxx_include_path.sh
export CPLUS_INCLUDE_PATH=$(get_cxx_include_path)

echo -e "Local branches:\n"
git branch

export SOURCE_DIR=`git rev-parse --show-toplevel`

function h () {
cat << EndOfMessage

To select conda environment, define:

  export USE_ENV=$USE_ENV

for instance, before sourcing this script.

To enable different CUDA version, say 11.0, run
  conda install -c conda-forge -c nvcc_linux-64=11.0  [Is this required??]
  conda deactivate
  export CUDA_VERSION=11.7.1  [currently CUDA_VERSION=${CUDA_VERSION}]
  <source the activate-heavydb-internal-dev.sh script>
  <clean & re-build>

To apply patches, run:

  patch -p1 < nvcc-boost-include-dirs.patch  [apply for heavydb 5.0]

To configure, run:

  mkdir -p build-nocuda && cd build-nocuda
  cmake -Wno-dev \$CMAKE_OPTIONS_NOCUDA ..

  mkdir -p build && cd build
  cmake -Wno-dev \$CMAKE_OPTIONS_CUDA ..

  mkdir -p build-nocuda-dbe && cd build-nocuda-dbe
  cmake -Wno-dev \$CMAKE_OPTIONS_NOCUDA_DBE ..

  mkdir -p build-dbe && cd build-cuda-dbe
  cmake -Wno-dev \$CMAKE_OPTIONS_CUDA_DBE ..

  mkdir -p build-nocuda-mlpack && cd build-nocuda-mlpack
  cmake -Wno-dev \$CMAKE_OPTIONS_NOCUDA_MLPACK ..

  mkdir -p build-tsan && cd build-tsan
  cmake -Wno-dev \$CMAKE_OPTIONS_TSAN ..

To build, run:

  make -j $NCORES

To test, run:

  mkdir -p tmp && bin/initheavy -f tmp
  make sanity_tests

To valgrind, run:

  cd Tests
  mkdir -p tmp && ../bin/initheavy -f tmp
  valgrind --suppressions=../../config/valgrind.suppressions --gen-suppressions=all \\
           --show-leak-kinds=definite --tool=memcheck \\
           --exit-on-first-error=yes --error-exitcode=777 \\
           --leak-check=full \\
           ./ExecuteTest --with-sharding

To test concurrency, run

  mkdir -p chaos && bin/initheavy -f chaos

  export TSAN_OPTIONS=suppressions=\$SOURCE_DIR/config/tsan.suppressions
  bin/heavyai_server --data chaos
  java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar com.mapd.tests.AlterDropTruncateValidateConcurrencyTest
  java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar com.mapd.tests.CatalogConcurrencyTest
  java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar com.mapd.tests.SelectUpdateDeleteDifferentTables

  export TSAN_OPTIONS=suppressions=\$SOURCE_DIR/config/tsan.suppressions:history_size=7
  bin/heavyai_server --data chaos
  java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar com.mapd.tests.EagainConcurrencyTest

  export TSAN_OPTIONS=suppressions=\$SOURCE_DIR/config/tsan.suppressions:history_size=7:halt_on_error=1
  bin/heavyai_server --data chaos
  java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar com.mapd.tests.ForeignStorageConcurrencyTest

  export TSAN_OPTIONS=suppressions=\$SOURCE_DIR/config/tsan.suppressions:second_deadlock_stack=true:detect_deadlocks=true:history_size=7:halt_on_error=1
  bin/heavyai_server --data chaos
  java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar com.mapd.tests.ForeignTableRefreshConcurrencyTest

  export TSAN_OPTIONS=suppressions=\$SOURCE_DIR/config/tsan.suppressions
  bin/heavyai_server --data chaos --allowed-import-paths '["/"]' --allowed-export-paths '["/"]'
  java -Dfile.encoding=UTF-8 -cp ../java/utility/target/utility-1.0-SNAPSHOT-jar-with-dependencies.jar com.mapd.tests.ImportAlterValidateSelectConcurrencyTest ../Tests/Import/datafiles/mixed_varlen.txt ../Tests/Import/datafiles/geospatial_mpoly/geospatial_mpoly.shp

To serve, run:

  mkdir -p data && bin/initheavy -f data
  bin/heavydb --data data --enable-dev-table-functions --enable-udf-registration-for-all-users

Use the following server options as needed (see \`bin/heavydb --help\` for details):

  --udf ../Tests/Udf/device_selection_samples.cpp \\
  --log-channels PTX,IR \\
  --log-severity-clog=WARNING \\
  --num-executors=4

To try out sql commands, serve (see above) and run:

  echo "select * from table (generate_series(1, 5));" | bin/heavysql -p HyperInteractive -u admin

EndOfMessage

}

h
