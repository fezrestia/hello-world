#include <fstream>
#include <iostream>

#include "util.h"

using namespace std;

namespace fezrestia {

FileBuffer* loadFile(const char* fileName) {
    // Open.
    ifstream inputFile(fileName, ifstream::binary);

    // Seek to stop end.
    inputFile.seekg(0, ifstream::end);

    // Get file size.
    int fileSize = static_cast<int> (inputFile.tellg());

    // Seek to start end.
    inputFile.seekg(0, ifstream::beg);

    // Alloc buffer.
    char* fileBuf = new char[fileSize];

    // Load.
    inputFile.read(fileBuf, fileSize);

    // FileBuffer instance.
    FileBuffer* fileBuffer = new FileBuffer(fileSize, fileBuf);

    return fileBuffer;
}

} // namespace fezrestia
