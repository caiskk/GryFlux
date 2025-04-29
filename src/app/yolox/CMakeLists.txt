add_library(app_includes INTERFACE)

find_package(OpenCV REQUIRED)

target_include_directories(app_includes INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/sink
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_SOURCE_DIR}/package
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
)
set(rknn_lib /usr/lib/librknnrt.so)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/package ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty ${OpenCV_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/sink/consumer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/source/producer APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/image_preprocess APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/rk_runner APP_SRC)
# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/object_tracker APP_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/tasks/res_sender APP_SRC)

add_executable(yolox_stream yolox_stream.cpp ${SRC_DIR} ${APP_SRC})

target_link_libraries(yolox_stream ${app_includes} ${dynamic_libs} ${OpenCV_LIBS} ${rknn_lib})
install(TARGETS yolox_stream RUNTIME DESTINATION ./)
