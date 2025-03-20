add_library(app_includes INTERFACE)

target_include_directories(app_includes INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/sink
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_SOURCE_DIR}/package
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/package)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/sink/test_consumer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/source/test_producer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/feature_extractor APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/image_preprocess APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/object_detector APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/object_tracker APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/res_sender APP_SRC)

add_executable(example_stream example_stream.cpp ${SRC_DIR} ${APP_SRC})

target_link_libraries(example_stream ${app_includes} ${dynamic_libs} )
install(TARGETS example_stream RUNTIME DESTINATION ./)
