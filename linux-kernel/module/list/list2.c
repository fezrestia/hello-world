#include <linux/module.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/debugfs.h>

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Aki.SHIMO. <fezrestia@gmail.com>");
MODULE_DESCRIPTION("Stack Sample");

static LIST_HEAD(teststack);

struct teststack_entry {
    struct list_head list;
    int n;
};

static void teststack_push(int n)
{
    struct teststack_entry* e = kmalloc(sizeof(*e), GFP_KERNEL);
    e->n = n;
    list_add(&e->list, &teststack);
}

static int teststack_pop(int* np)
{
    struct teststack_entry* e;

    if (list_empty(&teststack)) {
        return -1;
    }

    e = list_first_entry(&teststack, struct teststack_entry, list);

    if (np != NULL) {
        *np = e->n;
    }

    list_del(&e->list);
    kfree(e);

    return 0;
}

static void teststack_clean_out(void)
{
    while (!list_empty(&teststack)) {
        teststack_pop(NULL);
    }
}

static struct dentry* teststack_dir;
static struct dentry* showfile;
static struct dentry* pushfile;
static struct dentry* popfile;

static char testbuf[1024];

static ssize_t show_read(struct file* f, char* __user buf, size_t len, loff_t* ppos)
{
    char* bufp = testbuf;
    size_t remain = sizeof(testbuf);
    struct teststack_entry* e;
    size_t l;

    if (list_empty(&teststack)) {
        return simple_read_from_buffer(buf, len, ppos, "\n", 1);
    }

    list_for_each_entry(e, &teststack, list) {
        int n;

        n = snprintf(bufp, remain, "%d ", e->n);
        if (n == 0) {
            break;
        }
        bufp += n;
        remain -= n;
    }

    l = strlen(testbuf);
    testbuf[l - 1] = '\n';

    return simple_read_from_buffer(buf, len, ppos, testbuf, l);
}

static ssize_t push_write(struct file* f, const char* __user buf, size_t len, loff_t* ppos)
{
    ssize_t ret;
    int n;

    ret = simple_write_to_buffer(testbuf, sizeof(testbuf), ppos, buf, len);

    if (ret < 0) {
        return ret;
    }

    sscanf(testbuf, "%20d", &n);
    teststack_push(n);

    return ret;
}

static ssize_t pop_read(struct file* f, char* __user buf, size_t len, loff_t* ppos)
{
    int n;

    if (*ppos || teststack_pop(&n) == -1) {
        return 0;
    }

    snprintf(testbuf, sizeof(testbuf), "%d\n", n);

    return simple_read_from_buffer(buf, len, ppos, testbuf, strlen(testbuf));
}

static struct file_operations show_fops = {
    .owner = THIS_MODULE,
    .read = show_read,
};

static struct file_operations push_fops = {
    .owner = THIS_MODULE,
    .write = push_write,
};

static struct file_operations pop_fops = {
    .owner = THIS_MODULE,
    .read = pop_read,
};

static int mod_init(void)
{
    teststack_dir = debugfs_create_dir("teststack", NULL);
    if (!teststack_dir) {
        return -ENOMEM;
    }
    showfile = debugfs_create_file("show", 0400, teststack_dir, NULL, &show_fops);
    if (!showfile) {
        goto fail;
    }
    pushfile = debugfs_create_file("push", 0200, teststack_dir, NULL, &push_fops);
    if (!pushfile) {
        goto fail;
    }

    popfile = debugfs_create_file("pop", 0400, teststack_dir, NULL, &pop_fops);
    if (!popfile) {
        goto fail;
    }

    return 0;

fail:
    debugfs_remove_recursive(teststack_dir);
    return -ENOMEM;
}

static void mod_exit(void)
{
    debugfs_remove_recursive(teststack_dir);
    teststack_clean_out();
}

module_init(mod_init);
module_exit(mod_exit);

