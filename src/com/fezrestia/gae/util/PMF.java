package com.fezrestia.gae.util;

import javax.jdo.JDOHelper;
import javax.jdo.PersistenceManagerFactory;

public final class PMF {
    private static final PersistenceManagerFactory PMF_INSTANCE
            = JDOHelper.getPersistenceManagerFactory("transactions-optional");

    /**
     * CONSTRUCTOR.
     */
    private PMF() {
        // NOP.
    }

    public static PersistenceManagerFactory get() {
        return PMF_INSTANCE;
    }
}
