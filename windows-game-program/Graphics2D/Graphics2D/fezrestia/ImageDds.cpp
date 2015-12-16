// Implementation for
#include "ImageDds.h"

// Depends on
#include "FileBuffer.h"
#include "BinaryLoader.h"

namespace fezrestia {

ImageDds::ImageDds(const char* fileName) :
        mWidth(0),
        mHeight(0),
        mImage(NULL) {
    // Load file.
    FileBuffer* fileBuf = new FileBuffer(fileName);

    // Size.
    mWidth = BinaryLoader::getUnsignedIntAsLittleEndianFrom(fileBuf->getBuffer() + 12);
    mHeight = BinaryLoader::getUnsignedIntAsLittleEndianFrom(fileBuf->getBuffer() + 16);

    // Data.
    mImage = new unsigned int[mWidth * mHeight];
    for (unsigned int i = 0; i < (mWidth * mHeight); ++i) {
        mImage[i] = BinaryLoader::getUnsignedIntAsLittleEndianFrom(
                fileBuf->getBuffer() + 128 + (i * 4));
    }

    // Release.
    fileBuf->release();
    delete fileBuf;
    fileBuf = NULL;
}

ImageDds::~ImageDds() {
    delete[] mImage;
    mImage = NULL;
}

unsigned int ImageDds::getWidth() {
    return mWidth;
}

unsigned int ImageDds::getHeight() {
    return mHeight;
}

void ImageDds::render(
        int srcLeft,
        int srcTop,
        int srcRight,
        int srcBottom,
        unsigned int* dstBuf,
        unsigned int dstBufWidth,
        unsigned int dstBufHeight,
        int dstLeft,
        int dstTop) {
    // Scan range.
    unsigned int rangeX = srcRight - srcLeft;
    unsigned int rangeY = srcBottom - srcTop;

    // Values.
    int srcPos = 0;
    int dstPos = 0;
    unsigned int srcA = 0;
    unsigned int srcR = 0;
    unsigned int srcG = 0;
    unsigned int srcB = 0;
    unsigned int dstR = 0;
    unsigned int dstG = 0;
    unsigned int dstB = 0;
    unsigned int resultR = 0;
    unsigned int resultG = 0;
    unsigned int resultB = 0;
    unsigned int resultColor = 0;

    for (unsigned int y = 0; y < rangeY; ++y) {
        for (unsigned int x = 0; x < rangeX; ++x) {
            // Check source buffer range.
            if (((srcLeft + x) < 0) || (mWidth <= (srcLeft + x))
                        || ((srcTop + y) < 0) || (mHeight <= (srcTop + y))) {
                // Target source buffer is out of range.
                continue;
            }
            // Check destination buffer range.
            if (((dstLeft + x) < 0) || (dstBufWidth <= (dstLeft + x))
                    || ((dstTop + y) < 0) || (dstBufHeight <= (dstTop + y))) {
                // Target destination buffer is out of range.
                continue;
            }

            srcPos = (srcTop + y) * mWidth + (srcLeft + x);
            dstPos = (dstTop + y) * dstBufWidth + (dstLeft + x);

            // Alpha. (0 - 255)
            srcA = (mImage[srcPos] & 0xFF000000) >> 24;

            // Source channel. (0 - 255)
            srcR = mImage[srcPos] & 0x00FF0000;
            srcG = mImage[srcPos] & 0x0000FF00;
            srcB = mImage[srcPos] & 0x000000FF;

            // Destination channel. (0 - 255)
            dstR = dstBuf[dstPos] & 0x00FF0000;
            dstG = dstBuf[dstPos] & 0x0000FF00;
            dstB = dstBuf[dstPos] & 0x000000FF;

            // Blend.
            resultR = (srcR - dstR) * srcA / 255 + dstR;
            resultG = (srcG - dstG) * srcA / 255 + dstG;
            resultB = (srcB - dstB) * srcA / 255 + dstB;
            
            resultColor = (resultR & 0xFF0000) | (resultG & 0x00FF00) | resultB;

            dstBuf[dstPos] = resultColor;
        }
    }
}

} // namespace fezrestia
