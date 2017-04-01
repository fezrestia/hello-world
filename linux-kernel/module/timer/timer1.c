#include <linux/module.h>
#include <linux/timer.h>

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Aki.SHIMO. <fezrestia@gmail.com>");
MODULE_DESCRIPTION("Timer Test Module");

struct timer_data {
    char *name;
    int interval;
    struct timer_list timer;
};

struct timer_data td[2] = {
    {
        .name = "foo",
        .interval = 2,
    },
    {
        .name = "bar",
        .interval = 3,
    },
};

#define TIMER_TIMEOUT_SECS      10

static void timer_func(unsigned long arg)
{
    struct timer_data* data = (struct timer_data*) arg;

    printk(KERN_ALERT "Timer:%s : %d sec DONE\n", data->name, data->interval);

    mod_timer(&data->timer, jiffies + data->interval * HZ);
}

static int mod_init(void)
{
    int i;
    for (i = 0; i < 2; ++i) {
        struct timer_data* d = &td[i];
        init_timer(&d->timer);
        d->timer.function = timer_func;
        d->timer.expires = jiffies + d->interval * HZ;
        d->timer.data = (unsigned long) d;
        add_timer(&d->timer);
    }

    return 0;
}

static void mod_exit(void)
{
    int i;
    for (i = 0; i < 2; ++i) {
        del_timer(&td[i].timer);
    }
}

module_init(mod_init);
module_exit(mod_exit);

