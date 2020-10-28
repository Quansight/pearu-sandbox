#pragma once

#ifndef __CUDACC__

#include <iostream>
#include <cxxabi.h>
#include <sstream>
#include <type_traits>
#include <vector>
#include <unordered_map>

template <typename T>
std::string typeName(const T* v) {
  std::stringstream stream;
  int status;
  char* demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
  stream << std::string(demangled);
  free(demangled);
  return stream.str();
}

namespace {

template <typename T, typename = void>
struct has_toString : std::false_type {};
template <typename T>
struct has_toString<T, decltype(std::declval<T>().toString(), void())> : std::true_type {
};
template <class T>
inline constexpr bool has_toString_v = has_toString<T>::value;

template <typename T, typename = void>
struct get_has_toString : std::false_type {};
template <typename T>
struct get_has_toString<T, decltype(std::declval<T>().get()->toString(), void())>
    : std::true_type {};
template <class T>
inline constexpr bool get_has_toString_v = get_has_toString<T>::value;

template <typename T, typename = void>
struct has_to_string : std::false_type {};
template <typename T>
struct has_to_string<T, decltype(std::declval<T>().to_string(), void())> : std::true_type {
};
template <class T>
inline constexpr bool has_to_string_v = has_to_string<T>::value;

}  // namespace

template <typename T>
std::string toString(const T& v) {
  if constexpr (std::is_same_v<T, std::string>) {
    return "\"" + v + "\"";
  } else if constexpr (std::is_arithmetic_v<T>) {
    return std::to_string(v);
  } else if constexpr (has_toString_v<T>) {
    return v.toString();
  } else if constexpr (has_to_string_v<T>) {
    return v.to_string();
  } else if constexpr (get_has_toString_v<T>) {
    return v.get()->toString();
  } else if constexpr (std::is_same_v<T, void*>) {
    std::ostringstream ss;
    ss << std::hex << (uintptr_t)v;
    return "0x" + ss.str();
  } else if constexpr (std::is_pointer_v<T>) {
    return (v == NULL ? "NULL" : "&" + toString(*v));
  } else {
    return typeName(&v);
  }
}

template <typename T>
std::string toString(const std::pair<T, T>& v) {
  return "(" + toString(v.first) + ", " + toString(v.second) + ")";
}

template <typename T1, typename T2>
std::string toString(const std::pair<T1, T2>& v) {
  return "(" + toString(v.first) + ", " + toString(v.second) + ")";
}

template <typename T>
std::string toString(const std::vector<T>& v) {
  auto result = std::string("[");
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      result += ", ";
    }
    result += toString(v[i]);
  }
  result += "]";
  return result;
}

template <typename T1, typename T2>
std::string toString(const std::unordered_map<T1, T2>& v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto& p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

#endif
