set(TARGET llama-ptq-geometry)
add_executable(${TARGET} ptq-geometry.cpp)
target_link_libraries(${TARGET} PRIVATE llama ggml ggml-cuda ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
