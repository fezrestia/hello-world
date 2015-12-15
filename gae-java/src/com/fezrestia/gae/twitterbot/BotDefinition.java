package com.fezrestia.gae.twitterbot;

import java.util.ArrayList;
import java.util.List;

import javax.jdo.PersistenceManager;
import javax.jdo.Query;
import javax.jdo.annotations.IdGeneratorStrategy;
import javax.jdo.annotations.IdentityType;
import javax.jdo.annotations.PersistenceCapable;
import javax.jdo.annotations.Persistent;
import javax.jdo.annotations.PrimaryKey;

import twitter4j.auth.AccessToken;
import twitter4j.auth.RequestToken;

import com.fezrestia.gae.util.PMF;
import com.google.appengine.api.users.User;

@PersistenceCapable(identityType = IdentityType.APPLICATION, detachable ="true")
public class BotDefinition {

    @PrimaryKey
    @Persistent(valueStrategy = IdGeneratorStrategy.IDENTITY)
    private Long id;

    @Persistent
    private User owner;

    @Persistent
    private String word;

    @Persistent
    private List<String> tweets;

    @Persistent
    private long sinceId;

    @Persistent
    private String requestToken;

    @Persistent
    private String requestTokenSecret;

    @Persistent
    private String accessToken;

    @Persistent
    private String accessTokenSecret;

    @Persistent
    private String twitterAccount;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getOwner() {
        return owner;
    }

    public void setOwner(User owner) {
        this.owner = owner;
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public List<String> getTweets() {
        return tweets;
    }

    public void setTweets(List<String> tweets) {
        this.tweets = tweets;
    }

    public long getSinceId() {
        return sinceId;
    }

    public void setSinceId(long sinceId) {
        this.sinceId = sinceId;
    }

    public RequestToken getRequestToken() {
        return new RequestToken(this.requestToken, this.requestTokenSecret);
    }

    public void setRequestToken(RequestToken token) {
        this.requestToken = token.getToken();
        this.requestTokenSecret = token.getTokenSecret();
    }

    public AccessToken getAccessToken() {
        return new AccessToken(accessToken, accessTokenSecret);
    }

    public void setAccessToken(AccessToken token) {
        this.accessToken = token.getToken();
        this.accessTokenSecret = token.getTokenSecret();
    }

    public boolean hasAccessToken() {
        return accessToken != null;
    }

    public String getTwitterAccount() {
        return twitterAccount;
    }

    public void setTwitterAccount(String twitterAccount) {
        this.twitterAccount = twitterAccount;
    }

    /**
     * CONSTRUCTOR.
     */
    public BotDefinition(){
        this.tweets = new ArrayList<String>();
        this.sinceId = 0;
    }

    /**
     * CONSTRUCTOR.
     *
     * @param owner
     * @param word
     */
    public BotDefinition(User owner, String word) {
        this();
        this.owner = owner;
        this.word = word;
    }

    /**
     * Random pick up word from list.
     *
     * @return
     */
    public String pickTweet() {
        int i = (int)(Math.random() * tweets.size());
        return tweets.get(i);
    }

    /**
     * Get BodDefinition from user.
     *
     * @param user
     * @return
     */
    public static List<BotDefinition> getBotDefinition(User user) {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            Query query = pm.newQuery(BotDefinition.class);

            query.setFilter("owner == user");
            query.declareParameters(User.class.getName() + " user");
            List<BotDefinition> bots = (List<BotDefinition>) query.execute(user);
            pm.detachCopyAll(bots);
            return bots;
        } finally {
            if (pm != null && !pm.isClosed())
                pm.close();
        }
    }

    /**
     * Get BotDefinition from bot ID.
     *
     * @param botId
     * @return
     */
    public static BotDefinition getBotDefinition(long botId) {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            BotDefinition bot = pm.getObjectById(BotDefinition.class, new Long(botId));
            pm.detachCopy(bot);
            return bot;
        } finally {
            if (pm != null && !pm.isClosed())
                pm.close();
        }
    }

    /**
     * Update BotDefinition.
     *
     * @param botId
     * @param user
     * @param word
     * @param tweet0
     * @param tweet1
     * @param tweet2
     * @return
     */
    public static BotDefinition update(
            String botId,
            User user,
            String word,
            String tweet0,
            String tweet1,
            String tweet2) {
        BotDefinition bot = null;
        if (botId.trim().isEmpty()) {
            bot = new BotDefinition(user, word);
        } else {
            bot = getBotDefinition(Long.parseLong(botId));
            bot.word = word;
        }
        bot.tweets.clear();
        if (tweet0 != null && !tweet0.trim().isEmpty()) bot.tweets.add(tweet0);
        if (tweet1 != null && !tweet1.trim().isEmpty()) bot.tweets.add(tweet1);
        if (tweet2 != null && !tweet2.trim().isEmpty()) bot.tweets.add(tweet2);

        makePersistent(bot);
        return bot;
    }

    /**
     * Delete BotDefinition.
     *
     * @param bot
     */
    public static void removeBotDefinition(BotDefinition bot) {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            BotDefinition tmpBot = pm.getObjectById(BotDefinition.class, bot.getId());
            pm.deletePersistent(tmpBot);
        } finally {
            if (pm != null && !pm.isClosed())
                pm.close();
        }
    }

    /**
     * Get bot list.
     *
     * @return
     */
    public static List<BotDefinition> getBots() {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            Query query = pm.newQuery(BotDefinition.class);
            List<BotDefinition> bots = (List<BotDefinition>) query.execute();
            pm.detachCopyAll(bots);
            return bots;
        } finally {
            if (pm != null && !pm.isClosed())
                pm.close();
        }
    }

    /**
     * Store bot.
     *
     * @param bot
     */
    public static void makePersistent(BotDefinition bot) {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            pm.makePersistent(bot);
        } finally {
            if (pm != null && !pm.isClosed())
                pm.close();
        }
    }
}
