#ifndef FEZRESTIA_FLAGINTBIT
#define FEZRESTIA_FLAGINTBIT

namespace fezrestia {

class FlagIntBit {

public:
    /**
     * CONSTRUCTOR.
     */
    FlagIntBit() : mFlagBits(0) {
        // NOP.
    }

    /**
     * DESTRUCTOR.
     */
    ~FlagIntBit() {
        // NOP.
    }

    /**
     * Set flag bits valid (ON).
     */
    void setValid(int bitFlags) {
        mFlagBits |= bitFlags;
    }

    /**
     * Set flag bits invalid (OFF).
     */
    void setInvalid(int bitFlags) {
        mFlagBits &= ~bitFlags;
    }

    /**
     * Check flag bit is valid or not.
     *
     * @return Flag is valid or not
     */
    bool isValid(int bitFlag) {
        return ((mFlagBits & bitFlag) != 0);
    }

    /**
     * Reset all flag bits to invalid.
     */
    void clearAll() {
        mFlagBits = 0;
    }

private:
    // Flag bit.
    unsigned int mFlagBits;

};

} // namespace fezrestia

#endif // FEZRESTIA_FLAGINTBIT
