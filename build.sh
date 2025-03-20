mkdir -p build
cd build
cmake .. \
        -DCMAKE_BUILD_TYPE="Release" \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DBUILD_TEST=False  
make install -j$(nproc)

if [ $? != 0 ]; then
    echo "Error occurs in complaining."
    exit 1
fi
cd ..
