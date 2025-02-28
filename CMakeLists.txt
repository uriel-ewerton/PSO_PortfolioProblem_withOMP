cmake_minimum_required(VERSION 3.10)
project(PSO_PortfolioProblem_withOMP C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

if(OpenMP_C_FOUND)
    message(STATUS "OpenMP encontrado!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP não encontrado!")
endif()

add_executable(PSO_PortfolioProblem_withOMP main.c)

target_link_libraries(PSO_PortfolioProblem_withOMP OpenMP::OpenMP_C)
