package com.fezrestia.gae.util;

import java.util.HashMap;
import java.util.Map;

import javax.jdo.annotations.IdGeneratorStrategy;
import javax.jdo.annotations.IdentityType;
import javax.jdo.annotations.PersistenceCapable;
import javax.jdo.annotations.Persistent;
import javax.jdo.annotations.PrimaryKey;

import com.google.appengine.api.datastore.Blob;
import com.google.appengine.api.images.Image;
import com.google.appengine.api.images.ImagesServiceFactory;

@PersistenceCapable(identityType = IdentityType.APPLICATION, detachable="true")
public final class ImageEntity {
    private static Map<Integer, String> contentTypeMap = new HashMap<Integer, String>();
    static {
        contentTypeMap.put(Image.Format.BMP.ordinal(), "image/bmp");
        contentTypeMap.put(Image.Format.GIF.ordinal(), "image/gif");
        contentTypeMap.put(Image.Format.ICO.ordinal(), "image/x-icon");
        contentTypeMap.put(Image.Format.JPEG.ordinal(), "image/jpeg");
        contentTypeMap.put(Image.Format.PNG.ordinal(), "image/png");
        contentTypeMap.put(Image.Format.TIFF.ordinal(), "image/tiff");
    }

    @PrimaryKey
    @Persistent(valueStrategy = IdGeneratorStrategy.IDENTITY)
    private Long id;

    @Persistent
    private Blob blob;

    @Persistent
    private Integer formatOrdinal;

    @Persistent
    private String name;

    public void setId(Long id) {
        this.id = id;
    }

    public Long getId() {
        return id;
    }

    public void setBlob(Blob blob) {
        this.blob = blob;
    }

    public Blob getBlob() {
        return blob;
    }

    public void setFormatOrdinal(Integer formatOrdinal) {
        this.formatOrdinal = formatOrdinal;
    }

    public Integer getFormatOrdinal() {
        return formatOrdinal;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    /**
     * CONSTRUCTOR.
     *
     * @param bytes
     */
    public ImageEntity(byte[] bytes) {
        this.blob = new Blob(bytes);
        Image image = ImagesServiceFactory.makeImage(bytes);
        this.formatOrdinal = image.getFormat().ordinal();
    }

    /**
     * Get Blob content bytes.
     *
     * @return
     */
    public byte[] getBytes() {
        return blob.getBytes();
    }

    /**
     * Get Image.
     *
     * @return
     */
    public Image getImage() {
        return ImagesServiceFactory.makeImage(blob.getBytes());
    }

    /**
     * Get content type.
     *
     * @return
     */
    public String getContentType() {
        return contentTypeMap.get(this.formatOrdinal);
    }
}
