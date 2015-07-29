package com.fezrestia.gae.memo;

import java.io.IOException;
import java.util.List;
import java.util.TimeZone;

import javax.jdo.PersistenceManager;
import javax.jdo.Query;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.context.Context;

import com.fezrestia.gae.util.PMF;
import com.fezrestia.gae.velocity.Renderer;
import com.google.appengine.api.users.User;
import com.google.appengine.api.users.UserService;
import com.google.appengine.api.users.UserServiceFactory;

@SuppressWarnings("serial")
public class MainPage extends HttpServlet {
    static {
        TimeZone.setDefault(TimeZone.getTimeZone("JST"));
    }

    @Override
    public void doGet(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        // Get user.
        UserService userService = UserServiceFactory.getUserService();
        User user = userService.getCurrentUser(); // If user does not login yet, return null.

        PersistenceManager pm = null;

        try {
            pm = PMF.get().getPersistenceManager();

            // Query.
            Query query = pm.newQuery(Memo.class);
            query.setOrdering("date desc");
            query.declareParameters("com.google.appengine.api.users.User user_name");
            query.setFilter("author == user_name");

            List<Memo> memos = (List<Memo>) query.execute(user);

            Context context = new VelocityContext();
            context.put("memos",  memos);

            // Sign in/out URL.
            String signOutUrl = userService.createLogoutURL(req.getRequestURI());
            String signInUrl = userService.createLoginURL(req.getRequestURI());
            context.put("signOutUrl", signOutUrl);
            context.put("signInUrl", signInUrl);
            context.put("user", user);

            resp.setContentType("text/html");
            resp.setCharacterEncoding("utf-8");

            Renderer.render("WEB-INF/memoMainPage.vm", context, resp.getWriter());
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (pm != null && !pm.isClosed()) {
                pm.close();
            }
        }
    }
}
