package com.fezrestia.gae.util;

import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Logger;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.fileupload.FileItemIterator;
import org.apache.commons.fileupload.FileItemStream;
import org.apache.commons.fileupload.servlet.ServletFileUpload;

@SuppressWarnings("serial")
public class ImageUpload extends HttpServlet {
    public static final String TAG = ImageUpload.class.getSimpleName();
    private static final Logger logger = Logger.getLogger(TAG);

    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws IOException {

        try {
            ServletFileUpload upload = new ServletFileUpload();
            response.setContentType("text/plain");
            String redirectTo = request.getParameter("redirect_to");
            if (redirectTo == null) {
                redirectTo = "/imagetransformDataManager";
            }

            // Upload file.
            FileItemIterator iterator = upload.getItemIterator(request);
            while (iterator.hasNext()) {
                FileItemStream item = iterator.next();
                InputStream stream = item.openStream();

                if (item.isFormField()) {
                    logger.warning("Got a form field: " + item.getFieldName());
                } else {
                    logger.warning("Got an uploaded file: "
                            + item.getFieldName() + ", name = "
                            + item.getName());
                    // Generate image.
                    ImageUtil.storeImageToDataStore(stream, item.getName());
                }
            }

            response.sendRedirect(redirectTo);
        } catch (Exception ex) {
            throw new IOException(ex);
        }
    }
}
