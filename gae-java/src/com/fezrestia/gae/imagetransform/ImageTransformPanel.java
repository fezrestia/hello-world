package com.fezrestia.gae.imagetransform;

import java.io.IOException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.context.Context;

import com.fezrestia.gae.velocity.Renderer;

@SuppressWarnings("serial")
public class ImageTransformPanel extends HttpServlet {
    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response)
            throws IOException {
        String imageId = request.getParameter("imageId");

        Context context = new VelocityContext();
        context.put("imageId", imageId);
        context.put("redirect_to", request.getRequestURI());

        response.setContentType("text/html");
        response.setCharacterEncoding("utf-8");

        Renderer.render(
                "WEB-INF/imagetransformImageTransformPanel.vm",
                context,
                response.getWriter());
    }
}
