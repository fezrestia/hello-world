#include <linux/module.h>
#include <linux/timer.h>
#include <linux/debugfs.h>

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Aki.SHIMO. <fezrestia@gmail.com>");
MODULE_DESCRIPTION("Debugfs Sample");

static struct dentry* testfile;
static char testbuf[128];

struct timer_list timer;

#define TIMER_TIMEOUT_SECS      ((unsigned long) 1000)

static void timer_func(unsigned long arg)
{
    printk(KERN_ALERT "Timer : %lu sec passed.\n", TIMER_TIMEOUT_SECS);
}

static ssize_t timer_remain_millis_read(
        struct file* f,
        char* __user buf,
        size_t len,
        loff_t *ppos)
{
    unsigned long diff_millis;
    unsigned long now = jiffies;

    if (time_after(timer.expires, now)) {
        diff_millis = (timer.expires - now) * 1000 / HZ;
    } else {
        diff_millis = 0;
    }

    snprintf(testbuf, sizeof(testbuf), "%lu\n", diff_millis);

    return simple_read_from_buffer(buf, len, ppos, testbuf, strlen(testbuf));
}

static struct file_operations test_fops = {
    .owner = THIS_MODULE,
    .read = timer_remain_millis_read,
};

static int mod_init(void)
{
    init_timer(&timer);
    timer.function = timer_func;
    timer.expires = jiffies + TIMER_TIMEOUT_SECS * HZ;
    timer.data = 0;
    add_timer(&timer);

    testfile = debugfs_create_file(
            "timer_remain_millis",
            0400,
            NULL,
            NULL,
            &test_fops);
    if (!testfile) {
        return -ENOMEM;
    } else {
        return 0;
    }
}

static void mod_exit(void)
{
    debugfs_remove(testfile);
    del_timer(&timer);
}

module_init(mod_init);
module_exit(mod_exit);

