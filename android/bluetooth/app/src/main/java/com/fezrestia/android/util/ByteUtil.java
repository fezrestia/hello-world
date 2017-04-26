package com.fezrestia.android.util;

import java.io.UnsupportedEncodingException;

/**
 * Class for Byte related utilities.
 */
public class ByteUtil {

    /**
     * Connvert byte[] to UTF-8 String.
     *
     * @param data byte[]
     * @return Strinng
     */
    public static String byte2Utf8(byte[] data) {
        if (data == null) return null;

        try {
            return new String(data, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            return null;
        }
    }

    /**
     * Convert byte[] to hex-style String.
     *
     * @param data byte[]
     * @return Hex-style String
     */
    public static String byte2HexStr(byte[] data) {
        if (data == null) return null;

        String hexStr = "";
        for (byte b : data) {
            hexStr += Integer.toHexString(b & 0xFF);
        }
        return hexStr;
    }



}
