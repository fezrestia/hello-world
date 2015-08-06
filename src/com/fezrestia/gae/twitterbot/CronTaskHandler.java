package com.fezrestia.gae.twitterbot;

import java.io.IOException;
import java.util.logging.Logger;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.appengine.api.taskqueue.Queue;
import com.google.appengine.api.taskqueue.QueueFactory;
import com.google.appengine.api.taskqueue.TaskOptions;

@SuppressWarnings("serial")
public class CronTaskHandler extends HttpServlet {

    private static final Logger LOGGER = Logger.getLogger(CronTaskHandler.class.getSimpleName());

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        Queue queue = QueueFactory.getDefaultQueue();

        // Get bot, and execute it on TaskQueue.
        for (BotDefinition eachBot : BotDefinition.getBots()) {
            if (!eachBot.hasAccessToken()) {
                continue;
            }

            // Task.
            TaskOptions taskOptions = TaskOptions.Builder.withDefaults();
            taskOptions.url("/twitterbotBotHandler");
            taskOptions.param("botId", "" + eachBot.getId());

            // Request to execute task on TaskQueue.
            queue.add(taskOptions);

            LOGGER.info("Submitted a task for botId " + eachBot.getId());
        }
    }
}
