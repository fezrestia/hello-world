#ifndef FEZRESTIA_ARRAYXY
#define FEZRESTIA_ARRAYXY

#include <stdlib.h>

namespace fezrestia {

template<class T> class ArrayXY {

public:
    /**
     * CONSTRUCTOR.
     *
     * @param sizeX
     * @param sizeY
     */
    ArrayXY(int sizeX, int sizeY) :
            mArray(NULL),
            mSizeX(sizeX),
            mSizeY(sizeY) {
        // Alloc memory.
        mArray = new T[mSizeX * mSizeY];
    }

    /**
     * DESTRUCTOR.
     */
    ~ArrayXY() {
        // Release memory.
        delete[] mArray;
        mArray = NULL;
    }

    /**
     * Get element on x-y coordinate.
     *
     * @return
     */
    T& operator()(int x, int y) {
        return mArray[y * mSizeX + x];
    }

    /**
     * Get element on x-y coordinate.
     *
     * @return
     */
    const T& operator()(int x, int y) const {
        return mArray[y * mSizeX + x];
    }

private:
    // 1 axis array.
    T* mArray;

    // Size of X coordinate.
    const int mSizeX;

    // Size of Y coordinate.
    const int mSizeY;
};

} // namespace fezrestia

#endif // FEZRESTIA_ARRAYXY
