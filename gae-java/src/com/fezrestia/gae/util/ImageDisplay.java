package com.fezrestia.gae.util;

import java.io.IOException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@SuppressWarnings("serial")
public class ImageDisplay extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws IOException {
        String imageId = request.getParameter("imageId");
        String delete = request.getParameter("delete");

        if (delete != null && delete.equals("true")) {
            ImageUtil.deleteImageInDataStore(new Long(imageId));
            String redirectTo = request.getParameter("redirect_to");
            if (redirectTo == null) {
                redirectTo = "/";
            }
            response.sendRedirect(redirectTo);;
            return;
        }

        ImageEntity entity = ImageUtil.getImageEntityFromDataStore(new Long(imageId));

        if (entity == null) {
            response.setStatus(HttpServletResponse.SC_NOT_FOUND);
        } else {
            response.setContentType(entity.getContentType());
            response.getOutputStream().write(entity.getBytes());
        }
    }
}
