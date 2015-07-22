package com.fezrestia.gae.memo;

import java.io.IOException;
import java.util.Date;

import javax.jdo.PersistenceManager;
import javax.servlet.http.*;

import com.fezrestia.gae.util.PMF;

public class New extends HttpServlet {
    public void doPost(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        String content = req.getParameter("content");
        Memo memo = new Memo(content, new Date());

        PersistenceManager pm = PMF.get().getPersistenceManager();
        try {
            pm.makePersistent(memo);
        } finally {
            pm.close();
        }
        resp.sendRedirect("/memoMainPage");
    }
}
