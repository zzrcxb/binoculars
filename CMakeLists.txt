cmake_minimum_required(VERSION 3.1)

project(uarch-toolkit VERSION 0.1)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -D_GNU_SOURCE -O1")

execute_process(COMMAND bash -c "gcc -march=native -Q --help=target | grep march | \
                                 head -n 1 | tr -d '[:space:]' | cut -d'=' -f2"
                OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Native arch=${ARCH}")

execute_process(COMMAND cat /sys/devices/system/cpu/vulnerabilities/meltdown
                OUTPUT_VARIABLE PTI OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Meltdown mitigation: ${PTI}")
if(PTI MATCHES ".*PTI")
    message(STATUS "KPTI detected")
else()
    message(STATUS "KPTI not detected")
    add_definitions(-DNOPTI)
endif()

include_directories(include)
include_directories(PTEditor)

add_subdirectory(src)
