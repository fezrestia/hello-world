package com.fezrestia.gae.twitterbot;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import com.fezrestia.PrivateConstants;
import com.fezrestia.gae.util.ErrorPage;
import com.google.appengine.api.users.User;
import com.google.appengine.api.users.UserServiceFactory;

import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.auth.AccessToken;
import twitter4j.auth.RequestToken;

@SuppressWarnings("serial")
public class BotEditorHandler extends HttpServlet {

    private static final TwitterFactory TWITTER_FACTORY = new TwitterFactory();

    @Override
    public void init() throws ServletException {
        super.init();
        ConsumerKey.init(this);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        User user = UserServiceFactory.getUserService().getCurrentUser();

        String botId = request.getParameter("botId");
        String word = request.getParameter("word");
        String tweet0 = request.getParameter("tweet0");
        String tweet1 = request.getParameter("tweet1");
        String tweet2 = request.getParameter("tweet2");

        if (word == null || word.isEmpty() || tweet0 == null || tweet0.isEmpty()){
            ErrorPage.create(response, "Not information fullfilled.", "/twitterbotBotList");
            return;
        }

        BotDefinition bot = BotDefinition.update(botId, user, word, tweet0, tweet1, tweet2);

        if (!bot.hasAccessToken()) {
            // Not authorized yet.

            //TODO:
            bot.setAccessToken(new AccessToken(
                    PrivateConstants.TwitterTest.ACCESS_TOKEN,
                    PrivateConstants.TwitterTest.ACCESS_TOKEN_SECRET));
            bot.setTwitterAccount("cameradevtest");

            BotDefinition.makePersistent(bot);

            response.sendRedirect("/twitterbotBotList");


//            Twitter twitter = TWITTER_FACTORY.getInstance();
//            twitter.setOAuthConsumer(ConsumerKey.getKey(), ConsumerKey.getSecret());
//            twitter.setOAuthAccessToken(new AccessToken(
//                    PrivateConstants.TwitterTest.ACCESS_TOKEN,
//                    PrivateConstants.TwitterTest.ACCESS_TOKEN_SECRET));
//
//            try {
//                RequestToken requestToken = twitter.getOAuthRequestToken();
//                bot.setRequestToken(requestToken);
//                BotDefinition.makePersistent(bot);
//                long id = bot.getId();
//                HttpSession session = request.getSession();
//                session.setAttribute("requestingBotId", "" + id);
//                response.sendRedirect(requestToken.getAuthorizationURL());
//            } catch (TwitterException e) {
//                throw new ServletException(e);
//            }
        } else {
            // Already authorized.
            response.sendRedirect("/twitterbotBotList");
        }
    }
}
