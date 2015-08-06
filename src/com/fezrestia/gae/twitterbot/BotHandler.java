package com.fezrestia.gae.twitterbot;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import twitter4j.Query;
import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.conf.Configuration;
import twitter4j.conf.ConfigurationBuilder;

@SuppressWarnings("serial")
public class BotHandler extends HttpServlet {

    private static final Logger LOGGER = Logger.getLogger(BotHandler.class.getSimpleName());

    @Override
    public void init() throws ServletException {
        super.init();
        ConsumerKey.init(this);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        String botId = request.getParameter("botId");

        // Get BotDefinition.
        BotDefinition bot = BotDefinition.getBotDefinition(Long.parseLong(botId));

        // Initialize Twitter.
        Configuration config = new ConfigurationBuilder()
                .setOAuthConsumerKey(ConsumerKey.getKey())
                .setOAuthConsumerSecret(ConsumerKey.getSecret())
                .setOAuthAccessToken(bot.getAccessToken().getToken())
                .setOAuthAccessTokenSecret(bot.getAccessToken().getTokenSecret())
                .build();
        TwitterFactory factory = new TwitterFactory(config);
        Twitter twitter = factory.getInstance();

        // Bot process.
        try {
            // Search.
            Query twitterQuery = new Query(bot.getWord());
            twitterQuery.setSinceId(bot.getSinceId());

            long lastId = bot.getSinceId();

            QueryResult result = twitter.search(twitterQuery);
            for (Status eachStatus : result.getTweets()) {
//                if (eachStatus.getUser().getScreenName().equals(twitter.getScreenName())) {
//                    // Do not handle own tweet.
//                    continue;
//                }

                // Generate reply message.
                String replyMsg = bot.pickTweet();
//                String replyMsg = bot.pickTweet() + " QT @" + eachStatus.getUser().getName()
//                        + " " + eachStatus.getText();

                if (140 < replyMsg.length()) {
                    replyMsg = replyMsg.substring(0, 140);
                }

                LOGGER.info(replyMsg);

                // Upload to Twitter.
                twitter.updateStatus(replyMsg);
                lastId = Math.max(eachStatus.getId(), lastId);
            }

            // Store since ID.
            bot.setSinceId(lastId);
            BotDefinition.makePersistent(bot);
        } catch (TwitterException e) {
            LOGGER.log(Level.SEVERE, "Failed to twitter API call", e);
            throw new ServletException(e);
        }
    }
}
