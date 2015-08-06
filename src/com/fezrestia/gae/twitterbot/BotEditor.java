package com.fezrestia.gae.twitterbot;

import java.io.IOException;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.context.Context;

import com.fezrestia.gae.util.ErrorPage;
import com.fezrestia.gae.velocity.Renderer;

@SuppressWarnings("serial")
public class BotEditor extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        String botId = request.getParameter("botId");
        String word = "";
        String tweet0 = "";
        String tweet1 = "";
        String tweet2 = "";

        // Get BOT definition.
        if (botId != null && !botId.isEmpty()) {
            BotDefinition bot = BotDefinition.getBotDefinition(Long.parseLong(botId));
            if (bot == null) {
                ErrorPage.create(response, "BOT ID is not valid", "/twitterbotBotList");
                return;
            }

            // Check deletion.
            if (request.getParameter("delete") != null) {
                BotDefinition.removeBotDefinition(bot);
                response.sendRedirect("/twitterbotBotList");
                return;
            }

            word = bot.getWord();
            List<String> tweets = bot.getTweets();
            if (tweets.size() > 0) tweet0 = tweets.get(0);
            if (tweets.size() > 1) tweet1 = tweets.get(1);
            if (tweets.size() > 2) tweet2 = tweets.get(2);

        } else {
            botId = "";
        }

        // Rendering.
        Context context = new VelocityContext();
        context.put("botId", botId);
        context.put("word", word);
        context.put("tweet0", tweet0);
        context.put("tweet1", tweet1);
        context.put("tweet2", tweet2);

        response.setContentType("text/html");
        response.setCharacterEncoding("utf-8");

        Renderer.render("WEB-INF/twitterbotBotEditor.vm", context, response.getWriter());
    }
}
