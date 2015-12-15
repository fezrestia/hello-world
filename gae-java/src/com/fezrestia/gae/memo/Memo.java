package com.fezrestia.gae.memo;

import java.util.Date;

import javax.jdo.annotations.IdGeneratorStrategy;
import javax.jdo.annotations.IdentityType;
import javax.jdo.annotations.PersistenceCapable;
import javax.jdo.annotations.Persistent;
import javax.jdo.annotations.PrimaryKey;

import com.google.appengine.api.users.User;

@PersistenceCapable(identityType = IdentityType.APPLICATION)
public class Memo {
    @PrimaryKey
    @Persistent(valueStrategy = IdGeneratorStrategy.IDENTITY)
    private Long id;

    @Persistent
    private User author;

    @Persistent
    private String content;

    @Persistent
    private Date date;

    /**
     * CONSTRUCTOR.
     *
     * @param author
     * @param content
     * @param date
     */
    public Memo(User author, String content, Date date) {
        this.author = author;
        this.content = content;
        this.date = date;
    }

    /**
     * Get ID.
     *
     * @return
     */
    public Long getId() {
        return id;
    }

    /**
     * Set author user.
     *
     * @param author
     */
    public void setAuthor(User author) {
        this.author = author;
    }

    /**
     * Get author user.
     *
     * @return
     */
    public User getAuthor() {
        return author;
    }

    /**
     * Set content string.
     *
     * @param content
     */
    public void setContent(String content) {
        this.content = content;
    }

    /**
     * Get content string.
     *
     * @return
     */
    public String getContent() {
        return content;
    }

    /**
     * Set date.
     *
     * @param date
     */
    public void setDate(Date date) {
        this.date = date;
    }

    /**
     * Get date.
     *
     * @return
     */
    public Date getDate() {
        return date;
    }

}