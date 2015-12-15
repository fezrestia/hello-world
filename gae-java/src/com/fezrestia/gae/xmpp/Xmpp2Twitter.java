package com.fezrestia.gae.xmpp;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.conf.Configuration;
import twitter4j.conf.ConfigurationBuilder;

import com.fezrestia.PrivateConstants;
import com.google.appengine.api.xmpp.Message;
import com.google.appengine.api.xmpp.XMPPService;
import com.google.appengine.api.xmpp.XMPPServiceFactory;

@SuppressWarnings("serial")
public class Xmpp2Twitter extends HttpServlet {
    // Log.
    private static final Logger LOGGER = Logger.getLogger(Xmpp2Twitter.class.getSimpleName());

    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws IOException {

        // XMPP service.
        XMPPService xmpp = XMPPServiceFactory.getXMPPService();

        // Get message from request.
        Message message = xmpp.parseMessage(request);

        String xmppAddress = message.getFromJid().getId();
        LOGGER.info("xmppAddress = " + xmppAddress);

        // Check.
        if (!xmppAddress.startsWith(PrivateConstants.Xmpp.ACCOUNT)) {
            LOGGER.severe(xmppAddress + " does not match.");
            return;
        }

        // Post to twitter.
        Configuration config = new ConfigurationBuilder()
                .setOAuthConsumerKey(PrivateConstants.Twitter.CONSUMER_KEY)
                .setOAuthConsumerSecret(PrivateConstants.Twitter.CONSUMER_SECRET)
                .setOAuthAccessToken(PrivateConstants.Twitter.ACCESS_TOKEN)
                .setOAuthAccessTokenSecret(PrivateConstants.Twitter.ACCESS_TOKEN_SECRET)
                .build();

        Twitter twitter = new TwitterFactory(config).getInstance();

        try {
            twitter.updateStatus(message.getBody());
        } catch (TwitterException e) {
            e.printStackTrace();
            LOGGER.log(Level.SEVERE, "Twitter error", e);
        }
    }
}
