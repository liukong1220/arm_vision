cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT 
    img_tools.cpp
    math_tools.cpp
    logger.cpp
    exiter.cpp
    crc.cpp
    recorder.cpp
)
