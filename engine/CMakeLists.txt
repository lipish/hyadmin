cmake_minimum_required(VERSION 3.15)
project(_ext)

add_subdirectory(csrc)
install(TARGETS _ext DESTINATION heyi)