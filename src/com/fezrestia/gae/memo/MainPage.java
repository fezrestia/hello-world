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
import com.fezrestia.gae.util.TransactionManager;
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
    public void doGet(
            HttpServletRequest request,
            HttpServletResponse response) throws IOException {
        // Get user.
        UserService userService = UserServiceFactory.getUserService();
        User user = userService.getCurrentUser(); // If user does not login yet, return null.

        PersistenceManager pm = null;

        try {

            pm = PMF.get().getPersistenceManager();

            TransactionManager.Process process = new LoadMainPageProcess(
                    request,
                    response,
                    userService,
                    user);

            TransactionManager.start(3, pm, process);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (pm != null && !pm.isClosed()) {
                pm.close();
            }
        }
    }

    private class LoadMainPageProcess implements TransactionManager.Process {
        private final HttpServletRequest mRequest;
        private final HttpServletResponse mResponse;

        private final UserService mUserService;
        private final User mUser;

        LoadMainPageProcess(
                HttpServletRequest request,
                HttpServletResponse response,
                UserService userService,
                User user) {
            mRequest = request;
            mResponse = response;
            mUserService = userService;
            mUser = user;
        }

        @Override
        public void process(PersistenceManager pm) {
            // Query.
            Query query = pm.newQuery(Memo.class);
            query.setOrdering("date desc");
            query.declareParameters("com.google.appengine.api.users.User user_name");
            query.setFilter("author == user_name");

            List<Memo> memos = (List<Memo>) query.execute(mUser);

            Context context = new VelocityContext();
            context.put("memos",  memos);

            // Sign in/out URL.
            String signOutUrl = mUserService.createLogoutURL(mRequest.getRequestURI());
            String signInUrl = mUserService.createLoginURL(mRequest.getRequestURI());
            context.put("signOutUrl", signOutUrl);
            context.put("signInUrl", signInUrl);
            context.put("user", mUser);

            mResponse.setContentType("text/html");
            mResponse.setCharacterEncoding("utf-8");

            try {
                Renderer.render("WEB-INF/memoMainPage.vm", context, mResponse.getWriter());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
