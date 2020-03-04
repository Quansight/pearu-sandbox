{ stdenv, fetchurl, fetchFromGitHub, fixDarwinDylibNames, autoconf, boost
, brotli, cmake, double-conversion, flatbuffers, gflags, glog, gtest, lz4
, python3, rapidjson, snappy, thrift, uriparser, zlib, zstd, ncurses }:

let
  # Enable non-bundled uriparser
  # Introduced in https://github.com/apache/arrow/pull/4092
  Finduriparser_cmake = fetchurl {
    url =
      "https://raw.githubusercontent.com/apache/arrow/af4f52961209a5f1b43a19483536285c957e3bed/cpp/cmake_modules/Finduriparser.cmake";
    sha256 = "1cylrw00n2nkc2c49xk9j3rrza351rpravxgpw047vimcw0sk93s";
  };
  
in stdenv.mkDerivation rec {
  name = "arrow-cpp-${version}";
  version = "0.13.0";

  src = fetchurl {
    url =
      "mirror://apache/arrow/arrow-${version}/apache-arrow-${version}.tar.gz";
    sha256 = "06irh5zx6lc7jjf6hpz1vzk0pvbdx08lcirc8cp8ksb8j7fpfamc";
  };

  sourceRoot = "apache-arrow-${version}/cpp";

  nativeBuildInputs = [
    cmake
    autoconf # for vendored jemalloc
  ] ++ stdenv.lib.optional stdenv.isDarwin fixDarwinDylibNames;
  buildInputs = [
    boost
    brotli
    double-conversion
    flatbuffers
    gflags
    glog
    gtest
    lz4
    rapidjson
    snappy
    thrift
    uriparser
    zlib
    zstd
    ncurses
    python3.pkgs.python
    python3.pkgs.numpy
  ];

  preConfigure = ''
    substituteInPlace cmake_modules/FindLz4.cmake --replace CMAKE_STATIC_LIBRARY CMAKE_SHARED_LIBRARY

    cp ${Finduriparser_cmake} cmake_modules/Finduriparser.cmake

    patchShebangs build-support/

    # Fix build for ARROW_USE_SIMD=OFF
    # https://jira.apache.org/jira/browse/ARROW-5007
    sed -i src/arrow/util/sse-util.h -e '1i#include "arrow/util/logging.h"'
    sed -i src/arrow/util/neon-util.h -e '1i#include "arrow/util/logging.h"'
  '';

  cmakeFlags = [
    "-DARROW_BUILD_TESTS=OFF"
    "-DARROW_DEPENDENCY_SOURCE=SYSTEM"
    "-DARROW_PARQUET=ON"
    "-DARROW_PYTHON=ON"
    "-Duriparser_SOURCE=SYSTEM"
  ] ++ stdenv.lib.optional (!stdenv.isx86_64) "-DARROW_USE_SIMD=OFF";

  doInstallCheck = false;
}
