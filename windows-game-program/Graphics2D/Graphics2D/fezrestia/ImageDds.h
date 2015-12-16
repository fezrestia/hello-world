#ifndef FEZRESTIA_IMAGEDDS_H
#define FEZRESTIA_IMAGEDDS_H

namespace fezrestia {

class ImageDds {

public:
    /**
     * CONSTRUCTOR.
     *
     * @param fileName
     */
    ImageDds(const char* fileName);

    /**
     * DESTRUCTOR.
     */
    ~ImageDds();

    /**
     * Get image width.
     */
    unsigned int getWidth();

    /**
     * Get image height.
     */
    unsigned int getHeight();

    /**
     * Render image to destination buffer.
     * Destination buffer color format is ARGB8888.
     * (1 pixel = 1 byte)
     *
     * @param srcLeft
     * @param srcTop
     * @param srcRight
     * @param srcBottom
     * @param dstBuf
     * @param dstBufWidth
     * @param dstBufHeight
     * @param dstLeft
     * @param dstTop
     */
    void render(
            int srcLeft,
            int srcTop,
            int srcRight,
            int srcBottom,
            unsigned int* dstBuf,
            unsigned int dstBufWidth,
            unsigned int dstBufHeight,
            int dstLeft,
            int dstTop);

private:
    // Image width.
    unsigned int mWidth;
    // Image height.
    unsigned int mHeight;
    // Image buffer.
    unsigned int* mImage;

};

} // namespace fezrestia

#endif // FEZRESTIA_IMAGEDDS_H
