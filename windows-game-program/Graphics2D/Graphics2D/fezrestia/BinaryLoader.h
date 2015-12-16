#ifndef FEZRESTIA_BINARYLOADER_H
#define FEZRESTIA_BINARYLOADER_H

namespace fezrestia {

class BinaryLoader {

public:
    /**
     * Get unsigned int value from pointer.
     *
     * @param p
     * @return
     */
    static unsigned int getUnsignedIntAsLittleEndianFrom(const char* p) {
        const unsigned char* up;
        up = reinterpret_cast<const unsigned char*>(p);

        unsigned int ret = up[0];

        ret |= (up[1] << 8);
        ret |= (up[2] << 16);
        ret |= (up[3] << 24);

        return ret;
    }
};

} // namespace fezrestia

#endif // FEZRESTIA_BINARYLOADER_H
