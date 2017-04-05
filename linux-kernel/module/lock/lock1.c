#include <linux/module.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/debugfs.h>

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Aki.SHIMO. <fezrestia@gmail.com>");
MODULE_DESCRIPTION("Stack Sample");

#define STACK_MAX_LEN 10

static DEFINE_MUTEX(stack_mutex);

static LIST_HEAD(teststack);
static int stack_len;

struct teststack_entry {
    struct list_head list;
    int n;
};

static void teststack_push(int n)
{
    struct teststack_entry* e;

    if (stack_len >= STACK_MAX_LEN) {
        return;
    }

    e = kmalloc(sizeof(*e), GFP_KERNEL);
    INIT_LIST_HEAD(&e->list);
    e->n = n;

    printk(KERN_ALERT "#### TestStack : list_add() : E");
    list_add(&e->list, &teststack);
    printk(KERN_ALERT "#### TestStack : list_add() : X");

    ++stack_len;
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

    printk(KERN_ALERT "#### TestStack : list_del() : E");
    list_del(&e->list);
    printk(KERN_ALERT "#### TestStack : list_del() : X");

    --stack_len;
    kfree(e);

    return 0;
}

static void teststack_clean_out(void)
{
    mutex_lock(&stack_mutex);

    while (!list_empty(&teststack)) {
        teststack_pop(NULL);
        --stack_len;
    }

    mutex_unlock(&stack_mutex);
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
    ssize_t ret;

    mutex_lock(&stack_mutex);

    if (list_empty(&teststack)) {
        ret = simple_read_from_buffer(buf, len, ppos, "\n", 1);
        mutex_unlock(&stack_mutex);
        return ret;
    }

    printk(KERN_ALERT "#### TestStack : list_for_each_entry() : E");
    list_for_each_entry(e, &teststack, list) {
        int n;

        n = snprintf(bufp, remain, "%d ", e->n);
        if (n == 0) {
            break;
        }
        bufp += n;
        remain -= n;
    }
    printk(KERN_ALERT "#### TestStack : list_for_each_entry() : X");

    l = strlen(testbuf);
    testbuf[l - 1] = '\n';

    ret = simple_read_from_buffer(buf, len, ppos, testbuf, l);

    mutex_unlock(&stack_mutex);
    return ret;
}

static ssize_t push_write(struct file* f, const char* __user buf, size_t len, loff_t* ppos)
{
    ssize_t ret;
    int n;

    mutex_lock(&stack_mutex);

    ret = simple_write_to_buffer(testbuf, sizeof(testbuf), ppos, buf, len);

    if (ret < 0) {
        mutex_unlock(&stack_mutex);
        return ret;
    }

    sscanf(testbuf, "%20d", &n);
    teststack_push(n);

    mutex_unlock(&stack_mutex);
    return ret;
}

static ssize_t pop_read(struct file* f, char* __user buf, size_t len, loff_t* ppos)
{
    ssize_t ret;
    int n;

    mutex_lock(&stack_mutex);

    if (*ppos || teststack_pop(&n) == -1) {
        mutex_unlock(&stack_mutex);
        return 0;
    }

    snprintf(testbuf, sizeof(testbuf), "%d\n", n);

    ret = simple_read_from_buffer(buf, len, ppos, testbuf, strlen(testbuf));

    mutex_unlock(&stack_mutex);
    return ret;
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

