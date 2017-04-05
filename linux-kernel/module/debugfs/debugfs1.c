#include <linux/module.h>
#include <linux/timer.h>
#include <linux/debugfs.h>

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Aki.SHIMO. <fezrestia@gmail.com>");
MODULE_DESCRIPTION("Debugfs Sample");

static struct dentry* testdir;
static struct dentry* testfile;
static char testbuf[128];

struct timer_list timer;

static unsigned long timer_timeout_millis = 1000 * 1000;

static void timer_func(unsigned long arg)
{
    printk(KERN_ALERT "Timer : %lu sec passed.\n", timer_timeout_millis / 1000);
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

static ssize_t timer_remain_millis_write(
        struct file* f,
        const char* __user buf,
        size_t len,
        loff_t* ppos)
{
    ssize_t ret;

    ret = simple_write_to_buffer(testbuf, sizeof(testbuf), ppos, buf, len);
    if (ret < 0) {
        return ret;
    }

    sscanf(testbuf, "%20lu", &timer_timeout_millis);

    mod_timer(&timer, jiffies + timer_timeout_millis / 1000 * HZ);

    return ret;
}

static struct file_operations test_fops = {
    .owner = THIS_MODULE,
    .read = timer_remain_millis_read,
    .write = timer_remain_millis_write,
};

static int mod_init(void)
{
    init_timer(&timer);
    timer.function = timer_func;
    timer.expires = jiffies + timer_timeout_millis / 1000 * HZ;
    timer.data = 0;
    add_timer(&timer);

    testdir = debugfs_create_dir("test_timer", NULL);
    if (!testdir) {
        return -ENOMEM;
    }
    testfile = debugfs_create_file(
            "timer_remain_millis",
            0600,
            testdir,
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
    debugfs_remove_recursive(testdir);
    del_timer(&timer);
}

module_init(mod_init);
module_exit(mod_exit);

