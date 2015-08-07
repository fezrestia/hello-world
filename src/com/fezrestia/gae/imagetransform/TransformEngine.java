package com.fezrestia.gae.imagetransform;

import java.io.IOException;
import java.text.MessageFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.fezrestia.gae.util.ImageEntity;
import com.fezrestia.gae.util.ImageUtil;
import com.google.appengine.api.images.CompositeTransform;
import com.google.appengine.api.images.Image;
import com.google.appengine.api.images.ImagesServiceFactory;
import com.google.appengine.api.images.Transform;

@SuppressWarnings("serial")
public class TransformEngine extends HttpServlet {
    public static final String TAG = TransformEngine.class.getSimpleName();
    private static final Logger LOGGER = Logger.getLogger(TAG);

    // Transform mapping.
    private static final Map<String, Transform> mTransformMap = new HashMap<String, Transform>();
    static {
        mTransformMap.put("rotate-90", ImagesServiceFactory.makeRotate(90));
        mTransformMap.put("rotate-180", ImagesServiceFactory.makeRotate(180));
        mTransformMap.put("rotate-270", ImagesServiceFactory.makeRotate(270));
        mTransformMap.put("flip-ud", ImagesServiceFactory.makeVerticalFlip());
        mTransformMap.put("flip-lr", ImagesServiceFactory.makeHorizontalFlip());
        mTransformMap.put("lucky", ImagesServiceFactory.makeImFeelingLucky());
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String redirectTo = request.getParameter("redirect_to");
        String imageId = request.getParameter("imageId");

        CompositeTransform ct = ImagesServiceFactory.makeCompositeTransform();

        // Generate transform, and add to composite transform.
        for (int i = 1; i <= 5; i++) {
            String ti = request.getParameter("t" + i);
            String tiText = request.getParameter("t" + i + "-text");
            Transform tr = getTransform(ti, tiText);
            if (tr != null) {
                ct.concatenate(tr);
            }
        }

        // Get ImageEntity.
        ImageEntity imageEntity = ImageUtil.getImageEntityFromDataStore(Long.parseLong(imageId));

        // Apply transform.
        Image newImage = ImagesServiceFactory.getImagesService().applyTransform(
                ct,
                imageEntity.getImage());

        // Output to DataStore.
        Long newImageId = ImageUtil.storeImageToDataStore(
                newImage.getImageData(),
                imageEntity.getName() + ".mod");

        if (redirectTo != null) {
            response.sendRedirect(redirectTo + "?imageId=" + newImageId);
        } else {
            response.sendRedirect("/imagetransformDataManager");
        }
    }

    private Transform getTransform(String tag, String param) {
        if (tag == null) {
            return null;
        }

        Transform t = mTransformMap.get(tag);

        if (t != null) {
            return t;
        }

        // Not mapped. (Resize or Crop)
        if (tag.equals("resize")) {
            String[] p = param.split("[,x ]");
            if (p.length != 2) {
                LOGGER.warning("size spec is illegal");
                throw new IllegalArgumentException();
            }
            return ImagesServiceFactory.makeResize(
                    Integer.parseInt(p[0].trim()),
                    Integer.parseInt(p[1].trim()));
        }
        if (tag.equals("crop")) {
            String[] p = param.split("[, ]");
            if (p.length != 4) {
                LOGGER.warning("size spec is illegal");
                throw new IllegalArgumentException();
            }
            double d[] = new double[4];
            for (int i = 0; i < 4; i++) {
                d[i] = Double.parseDouble(p[i].trim());
            }
            LOGGER.info(MessageFormat.format("crop - {0}, {1}, {2}, {3}",
                    d[0], d[1], d[2], d[3]));
            return ImagesServiceFactory.makeCrop(d[0], d[1], d[2], d[3]);
        }
        return null;
    }
}
