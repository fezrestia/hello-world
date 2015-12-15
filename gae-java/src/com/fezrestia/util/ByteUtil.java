package com.fezrestia.util;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public final class ByteUtil {

    /**
     * Get byte array from file.
     *
     * @param fileName
     * @return
     */
    public static final byte[] getBytesFrom(String fileName) {
        // I/O.
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        FileInputStream is = null;
        try {
            is = new FileInputStream(fileName);
        } catch (FileNotFoundException e) {
            // Failed to open/read file.

            e.printStackTrace();
            return null;
        }

        // Mid buffer.
        byte[] buf = new byte[1024];

        // Read.
        int count = 0;
        try {
            while (0 < (count = is.read(buf))) {
                os.write(buf, 0, count);
            }
        } catch (IOException e) {
            // Failed to read file.
            e.printStackTrace();
            os = null;
        } finally {
            // Close.
            try {
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return os.toByteArray();
    }
}
