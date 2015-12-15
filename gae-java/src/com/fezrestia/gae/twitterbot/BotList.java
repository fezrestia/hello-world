package com.fezrestia.gae.twitterbot;

import java.io.IOException;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.context.Context;

import com.fezrestia.gae.velocity.Renderer;
import com.google.appengine.api.users.User;
import com.google.appengine.api.users.UserServiceFactory;

@SuppressWarnings("serial")
public class BotList extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        User user = UserServiceFactory.getUserService().getCurrentUser();

        List<BotDefinition> bots = BotDefinition.getBotDefinition(user);

        response.setContentType("text/html");
        response.setCharacterEncoding("utf-8");

        Context context = new VelocityContext();
        context.put("user", user);
        context.put("bots", bots);

        Renderer.render("WEB-INF/twitterbotBotList.vm", context, response.getWriter());
    }
}
