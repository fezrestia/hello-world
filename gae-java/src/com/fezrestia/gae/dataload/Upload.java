package com.fezrestia.gae.dataload;

import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.appengine.api.blobstore.BlobKey;
import com.google.appengine.api.blobstore.BlobstoreService;
import com.google.appengine.api.blobstore.BlobstoreServiceFactory;

@SuppressWarnings("serial")
public class Upload extends HttpServlet {
    public static final String TAG = Upload.class.getSimpleName();
    private static final Logger LOGGER = Logger.getLogger(TAG);

    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        BlobstoreService blobstoreService = BlobstoreServiceFactory.getBlobstoreService();

        Map<String, BlobKey> blobs = blobstoreService.getUploadedBlobs(request);

        for (String eachFileTag : blobs.keySet()) {
            LOGGER.info("uploaded : " + eachFileTag
                    + ", key=" + blobs.get(eachFileTag).getKeyString());
        }

        response.sendRedirect("/dataloadDataManager");
    }
}
