package com.fezrestia.gae.dataload;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.context.Context;

import com.fezrestia.gae.velocity.Renderer;
import com.google.appengine.api.blobstore.BlobInfo;
import com.google.appengine.api.blobstore.BlobInfoFactory;
import com.google.appengine.api.blobstore.BlobstoreService;
import com.google.appengine.api.blobstore.BlobstoreServiceFactory;

@SuppressWarnings("serial")
public class DataManager extends HttpServlet {

    public void doGet(HttpServletRequest request, HttpServletResponse response)
            throws IOException {
        BlobstoreService blobstoreService = BlobstoreServiceFactory.getBlobstoreService();
        BlobInfoFactory blobInfoFactory = new BlobInfoFactory();

        String uploadUrl = blobstoreService.createUploadUrl("/dataloadUpload");

        // Blob list.
        List<BlobInfo> blobs = new ArrayList<BlobInfo>();
        Iterator<BlobInfo> iter = blobInfoFactory.queryBlobInfos();
        while (iter.hasNext()) {
            blobs.add(iter.next());
        }

        // Render.
        Context context = new VelocityContext();
        context.put("uploadUrl", uploadUrl);
        context.put("blobs",  blobs);

        response.setContentType("text/html");
        response.setCharacterEncoding("UTF-8");

        Renderer.render("WEB-INF/dataloadDataManager.vm", context, response.getWriter());
    }
}
