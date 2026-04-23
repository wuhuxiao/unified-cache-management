if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES "./case/*.cc")
    list(FILTER UCMSHARED_TEST_SOURCE_FILES EXCLUDE REGEX ".*/metrics_cpp_writer\\.cc$")
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        trans metrics infra_logger
        gtest_main gtest
    )
    gtest_discover_tests(ucmshared.test)

    find_package(Python COMPONENTS Interpreter REQUIRED)
    add_library(metrics_cpp_writer SHARED case/metrics/metrics_cpp_writer.cc)
    target_link_libraries(metrics_cpp_writer PRIVATE metrics)
    add_test(
        NAME metrics_cpp_python_bridge
        COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/case/metrics/test_cpp_python_bridge.py
                $<TARGET_FILE:metrics_cpp_writer>
                $<TARGET_FILE_DIR:ucmmetrics>
    )
endif()
