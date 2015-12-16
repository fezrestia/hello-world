#ifndef FEZRESTIA_FILEBUFFER_H
#define FEZRESTIA_FILEBUFFER_H

#include <stdlib.h>
#include <fstream>

using namespace std;

namespace fezrestia {

/**
 * File buffer class.
 */
class FileBuffer {

public:
    /**
     * CONSTRUCTOR.
     *
     * @param size
     * @param buffer
     */
    FileBuffer(int size, char* buffer) :
            mSize(size),
            mBuffer(buffer) {
        // NOP.
    }

    /**
     * CONSTRUCTOR.
     *
     * @param fileName
     */
    FileBuffer(const char* fileName) :
            mSize(0),
            mBuffer(NULL) {
        // Load.
        loadFile(fileName);
    }

    /**
     * DESTRUCTOR.
     */
    ~FileBuffer() {
        // Fail safe.
        release();
    }

    /**
     * Get buffer size.
     *
     * @return
     */
    int getSize() {
        return mSize;
    }

    /**
     * Get buffer.
     *
     * @return
     */
    const char* getBuffer() {
        return mBuffer;
    }

    /**
     * Release all internal resources.
     */
    void release() {
        if (mBuffer != NULL) {
            delete mBuffer;
            mBuffer = NULL;
        }
    }

private:
    int mSize;
    char* mBuffer;

    // Load file to buffer.
    void loadFile(const char* fileName) {
        // Open.
        ifstream inputFile(fileName, ifstream::binary);
        // Seek to stop end.
        inputFile.seekg(0, ifstream::end);
        // Get file size.
        mSize = static_cast<int> (inputFile.tellg());
        // Seek to start end.
        inputFile.seekg(0, ifstream::beg);
        // Alloc buffer.
        mBuffer = new char[mSize];
        // Load.
        inputFile.read(mBuffer, mSize);
    }
};

} // namespace fezrestia

#endif // FEZRESTIA_FILEBUFFER_H
