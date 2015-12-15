package com.fezrestia.gae.util;

import java.util.logging.Logger;

import javax.jdo.JDOCanRetryException;
import javax.jdo.PersistenceManager;
import javax.jdo.Transaction;

public final class TransactionManager {
    public static final String TAG = TransactionManager.class.getSimpleName();
    private static final Logger LOGGER = Logger.getLogger(TAG);

    /**
     * Transaction exception.
     */
    public static class InternalTransactionException extends Exception {
        public final Object payload;

        /**
         * CONSTRUCTOR.
         *
         * @param msg
         * @param payload
         */
        public InternalTransactionException(String msg, Object payload) {
            super(msg);
            this.payload = payload;
        }
    }

    /**
     * Transaction process interface.
     */
    public static interface Process {
        /**
         * Single transaction process.
         *
         * @param pm
         */
        void process(PersistenceManager pm);
    }

    /**
     * Do transaction process.
     *
     * @param retryCount
     * @param pm
     * @param process
     * @return success or not
     * @throws InternalTransactionException
     */
    public static boolean start(
            int retryCount,
            PersistenceManager pm,
            Process process) throws InternalTransactionException {
        if (pm == null || process == null) {
            // Argument is invalid.
            return false;
        }

        Transaction tx = null;

        try {
            while (0 <= --retryCount) {
                tx = pm.currentTransaction();
                tx.begin();

                process.process(pm);

                try {
                    tx.commit();
                    return true;
                } catch (JDOCanRetryException e) {
                    LOGGER.warning("Transaction commit FAILED.");
                }
            }
        } finally {
            if (tx != null && tx.isActive()) {
                tx.rollback();
            }
        }

        return false;
    }
}
