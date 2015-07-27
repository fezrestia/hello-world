package com.fezrestia.gae.memo;

import java.util.Date;

import javax.jdo.annotations.IdGeneratorStrategy;
import javax.jdo.annotations.IdentityType;
import javax.jdo.annotations.PersistenceCapable;
import javax.jdo.annotations.Persistent;
import javax.jdo.annotations.PrimaryKey;

@PersistenceCapable(identityType = IdentityType.APPLICATION)
public class Memo {
    @PrimaryKey
    @Persistent(valueStrategy = IdGeneratorStrategy.IDENTITY)
    private Long id;

    @Persistent
    private String content;

    @Persistent
    private Date date;

    public Memo(String content, Date date) {
        this.content = content;
        this.date = date;
    }

    // getter/setter
    public Long getId() {
        return id;
    }
    public String getContent() {
        return content;
    }
    public Date getDate() {
        return date;
    }
    public void setContent(String content) {
        this.content = content;
    }
    public void setDate(Date date) {
        this.date = date;
    }
}