#ifndef FEZRESTIA_UTIL
#define FEZRESTIA_UTIL

using namespace std;

namespace fezrestia {

/**
 * File buffer class.
 */
class FileBuffer {
    private:
        const int mSize;
        const char* mBuffer;

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
};

/**
 * Load file and return FileBuffer. FileBuffer must be deleted by client.
 *
 * @param fileName [IN]
 * @return FileBuffer
 */
FileBuffer* loadFile(const char* fileName);

} // namespace fezrestia

#endif // FEZRESTIA_UTIL
