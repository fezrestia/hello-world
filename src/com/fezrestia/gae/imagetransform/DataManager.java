package com.fezrestia.gae.imagetransform;

import java.io.IOException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.context.Context;

import com.fezrestia.gae.util.ImageUtil;
import com.fezrestia.gae.velocity.Renderer;

@SuppressWarnings("serial")
public class DataManager extends HttpServlet {
    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response)
            throws IOException {

        Context context = new VelocityContext();
        context.put("images", ImageUtil.getAllImageEntitiesInDataStore());

        response.setContentType("text/html");
        response.setCharacterEncoding("utf-8");

        Renderer.render("WEB-INF/imagetransformDataManager.vm", context, response.getWriter());
    }
}
