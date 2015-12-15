package com.fezrestia.gae.twitterbot;

import java.util.logging.Logger;

import javax.servlet.http.HttpServlet;

import com.fezrestia.PrivateConstants;

public class ConsumerKey {
    private static final Logger LOGGER = Logger.getLogger(ConsumerKey.class.getSimpleName());

    private static String key;
    private static String secret;

    /**
     * Load consumer key/secret from file.
     *
     * @param servlet
     */
    public static void init(HttpServlet servlet) {
        if (key != null) {
            return;
        }

        key = PrivateConstants.TwitterTest.CONSUMER_KEY;
        secret = PrivateConstants.TwitterTest.CONSUMER_SECRET;

        if (key == null || secret == null) {
            LOGGER.severe("Failed to get consumer keys.");
        }
    }

    public static String getKey() {
        return key;
    }

    public static String getSecret() {
        return secret;
    }
}
