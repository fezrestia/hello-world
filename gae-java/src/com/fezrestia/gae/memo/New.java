package com.fezrestia.gae.memo;

import java.io.IOException;
import java.util.Date;

import javax.jdo.PersistenceManager;
import javax.servlet.http.*;

import com.fezrestia.gae.util.PMF;
import com.google.appengine.api.users.User;
import com.google.appengine.api.users.UserService;
import com.google.appengine.api.users.UserServiceFactory;

public class New extends HttpServlet {
    public void doPost(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        // Get user input content.
        String content = req.getParameter("content");

        // Generate memo instance.
        UserService userService = UserServiceFactory.getUserService();
        User user = userService.getCurrentUser();
        Memo memo = new Memo(user, content, new Date());

        PersistenceManager pm = PMF.get().getPersistenceManager();
        try {
            pm.makePersistent(memo);
        } finally {
            pm.close();
        }
        resp.sendRedirect("/memoMainPage");
    }
}
