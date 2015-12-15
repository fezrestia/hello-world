package com.fezrestia.gae.twitterbot;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.auth.AccessToken;

@SuppressWarnings("serial")
public class AuthHandler extends HttpServlet {

    private static final Logger LOGGER = Logger.getLogger(AuthHandler.class.getSimpleName());

    private static final TwitterFactory TWITTER_FACTORY = new TwitterFactory();

    @Override
    public void init() throws ServletException {
        super.init();

        // Load consumer key.
        ConsumerKey.init(this);
    }

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        // Get authorizing bot ID.
        HttpSession session = request.getSession();
        String idString = (String) session.getAttribute("requestingBotId");
        if (idString == null) {
            LOGGER.warning("Can not get requestingBotId");
            throw new ServletException("Can not get requestingBotId");
        }

        BotDefinition bot = BotDefinition.getBotDefinition(Long.parseLong(idString));
        Twitter twitter = TWITTER_FACTORY.getInstance();
        twitter.setOAuthConsumer(ConsumerKey.getKey(), ConsumerKey.getSecret());
        try {
            // Get access token.
            AccessToken accessToken = twitter.getOAuthAccessToken(bot.getRequestToken());
            bot.setAccessToken(accessToken);
            bot.setTwitterAccount(accessToken.getScreenName());
            BotDefinition.makePersistent(bot);
        } catch (TwitterException e) {
            LOGGER.log(Level.SEVERE, "Failed to get AccessToken", e);
            throw new ServletException("Failed to get AccessToken");
        }

        response.sendRedirect("/twitterbotBotList");
    }
}
