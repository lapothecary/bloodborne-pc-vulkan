cmake_minimum_required(VERSION 3.20)
project(bb_launcher_v2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

set(ATRAC9_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/LibAtrac9/C/src")
set(ATRAC9_SOURCES
    "${ATRAC9_DIR}/band_extension.c"
    "${ATRAC9_DIR}/bit_allocation.c"
    "${ATRAC9_DIR}/bit_reader.c"
    "${ATRAC9_DIR}/decinit.c"
    "${ATRAC9_DIR}/decoder.c"
    "${ATRAC9_DIR}/huffCodes.c"
    "${ATRAC9_DIR}/imdct.c"
    "${ATRAC9_DIR}/libatrac9.c"
    "${ATRAC9_DIR}/quantization.c"
    "${ATRAC9_DIR}/scale_factors.c"
    "${ATRAC9_DIR}/tables.c"
    "${ATRAC9_DIR}/unpack.c"
    "${ATRAC9_DIR}/utility.c"
)

add_library(bb_atrac9 STATIC ${ATRAC9_SOURCES})
target_include_directories(bb_atrac9 PUBLIC "${ATRAC9_DIR}")
if(MSVC)
    target_compile_options(bb_atrac9 PRIVATE /O2 /MT /W0)
else()
    target_compile_options(bb_atrac9 PRIVATE -O2 -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare)
endif()

set(LAUNCHER_SOURCES
    main.cpp
    launcher_bridge.cpp
    fsb_ui_sfx.cpp
)

if(WIN32)
    list(APPEND LAUNCHER_SOURCES launcher.rc)
    add_executable(Launcher WIN32 ${LAUNCHER_SOURCES})
    target_link_libraries(Launcher PRIVATE bb_atrac9)
    if(MSVC)
        target_compile_options(Launcher PRIVATE /O2 /MT /utf-8 /EHsc)
        target_link_options(Launcher PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
    endif()
    set(WEBVIEW2_SDK_DIR "" CACHE PATH "Path to unpacked Microsoft.Web.WebView2 NuGet package")
    if(WEBVIEW2_SDK_DIR)
        target_include_directories(Launcher PRIVATE "${WEBVIEW2_SDK_DIR}/build/native/include")
        target_link_libraries(Launcher PRIVATE "${WEBVIEW2_SDK_DIR}/build/native/x64/WebView2LoaderStatic.lib")
    endif()
    target_link_libraries(Launcher PRIVATE dwmapi shlwapi ole32 shell32 user32 gdi32 advapi32 uuid version)
else()
    add_executable(bb-launcher ${LAUNCHER_SOURCES})
    target_link_libraries(bb-launcher PRIVATE bb_atrac9)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(WEBKIT2GTK REQUIRED IMPORTED_TARGET webkit2gtk-4.1)
    target_link_libraries(bb-launcher PRIVATE PkgConfig::WEBKIT2GTK m)
endif()
