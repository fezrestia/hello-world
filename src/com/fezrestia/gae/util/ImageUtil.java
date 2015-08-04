package com.fezrestia.gae.util;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import javax.jdo.PersistenceManager;
import javax.jdo.Query;

import com.google.appengine.api.images.Image;
import com.google.appengine.api.images.ImagesServiceFactory;

public final class ImageUtil {

    /**
     * Store image bytes to DataStore.
     *
     * @param bytes
     * @return
     */
    public static Long storeImageToDataStore(byte[] bytes) {
        return storeImageToDataStore(bytes, null);
    }

    /**
     * Store image bytes to DataStore.
     *
     * @param bytes
     * @param name
     * @return
     */
    public static Long storeImageToDataStore(byte[] bytes, String name) {
        PersistenceManager pm = PMF.get().getPersistenceManager();
        ImageEntity entity = new ImageEntity(bytes);
        if (name != null) {
            entity.setName(name);
        }
        try {
            pm.makePersistent(entity);
        } finally {
            pm.close();
        }
        return entity.getId();
    }

    /**
     * Store stream image to DataStore.
     *
     * @param is
     * @param name
     * @return
     * @throws IOException
     */
    public static Long storeImageToDataStore(InputStream is, String name) throws IOException {
        return storeImageToDataStore(readBytes(is), name);
    }

    /**
     * Get ImageEntity from ID.
     *
     * @param imageId
     * @return
     */
    public static ImageEntity getImageEntityFromDataStore(Long imageId) {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            Query query = pm.newQuery(ImageEntity.class);

            query.setFilter("id == imageId");
            query.declareParameters("Long imageId");
            List<ImageEntity> images = (List<ImageEntity>) query.execute(new Long(imageId));
            if (!images.isEmpty()) {
                return images.get(0);
            } else {
                return null;
            }
        } finally {
            if (pm != null && !pm.isClosed()) {
                pm.close();
            }
        }
    }

    /**
     * Get all ImageEntity.
     *
     * @return
     */
    public static List<ImageEntity> getAllImageEntitiesInDataStore() {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            Query query = pm.newQuery(ImageEntity.class);
            List<ImageEntity> result = (List<ImageEntity>) query.execute();
            pm.detachCopyAll(result);
            return result;
        } finally {
            if (pm != null && !pm.isClosed()) {
                pm.close();
            }
        }
    }

    /**
     * Get Image from file.
     *
     * @param fileName
     * @return
     * @throws IOException
     */
    public static Image readImage(String fileName) throws IOException {
        File file = new File(fileName);
        InputStream is = new FileInputStream(file);
        int length = (int) file.length();
        byte[] buffer = new byte[length];

        int current = 0;
        while (current < length) {
            int len = is.read(buffer, current, length - current);
            if (len < 0) {
                is.close();
                throw new IOException("Could not fully read the file");
            }
            current += len;
        }

        is.close();

        return ImagesServiceFactory.makeImage(buffer);
    }

    /**
     * Delete ImageEntity.
     *
     * @param imageId
     */
    public static void deleteImageInDataStore(Long imageId) {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            Query query = pm.newQuery(ImageEntity.class);
            query.setFilter("id == imageId");
            query.declareParameters("Long imageId");
            List<ImageEntity> images = (List<ImageEntity>) query.execute(imageId);
            if (!images.isEmpty()) {
                pm.deletePersistent(images.get(0));
            }
        } finally {
            if (pm != null && !pm.isClosed()) {
                pm.close();
            }
        }
    }

    /**
     * Get bytes from InputStream.
     *
     * @param is
     * @return
     * @throws IOException
     */
    public static byte[] readBytes(InputStream is) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        int len;
        byte[] buf = new byte[1024];
        while (0 <= (len = is.read(buf))) {
            bos.write(buf, 0, len);
        }
        return bos.toByteArray();
    }
}
