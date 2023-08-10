cmake_minimum_required (VERSION 3.12)

project ("yolo-nas-deepsparse-cpp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PythonInterp 3.8 REQUIRED)

# Create a temporary Python script file
file(WRITE ${CMAKE_BINARY_DIR}/tmp_script.py
     "try:\n"
     "    import deepsparse\n"
     "    print(deepsparse.__path__[0] + ',' + deepsparse.cpu_architecture()['isa'])\n"
     "except Exception:\n"
     "    print('FAILED')\n")

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/tmp_script.py
    OUTPUT_VARIABLE PYTHON_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Clean up the temporary script file
file(REMOVE ${CMAKE_BINARY_DIR}/tmp_script.py)

if (PYTHON_OUTPUT STREQUAL "FAILED")
    message(FATAL_ERROR "Failed to determine deepsparse path. Make sure deepsparse is installed and it is supported by your hardware.")
else()

    # Split the Python output into PACKAGE_PATH and ARCH variables
    string(REPLACE "," ";" PYTHON_OUTPUT_LIST ${PYTHON_OUTPUT})
    list(GET PYTHON_OUTPUT_LIST 0 PACKAGE_PATH)
    list(GET PYTHON_OUTPUT_LIST 1 ARCH)

    file(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
    add_executable(${PROJECT_NAME} ${SOURCES})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    target_link_libraries(${PROJECT_NAME} argparse)
    target_link_directories(${PROJECT_NAME} PUBLIC "${PACKAGE_PATH}/${ARCH}/")
    target_link_libraries(${PROJECT_NAME} deepsparse)
    target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include")

    set(CPACK_PROJECT_NAME ${PROJECT_NAME})
    set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
    include(CPack)
endif()
