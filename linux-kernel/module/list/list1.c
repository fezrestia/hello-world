#include <linux/module.h>
#include <linux/slab.h>
#include <linux/list.h>

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Aki.SHIMO. <fezrestia@gmail.com>");
MODULE_DESCRIPTION("List Sample");

static LIST_HEAD(testlist);

struct testlist_entry {
    struct list_head list;
    int n;
};

static void testlist_add(int n) {
    struct testlist_entry* e = kmalloc(sizeof(*e), GFP_KERNEL);
    e->n = n;
    list_add(&e->list, &testlist);
    printk(KERN_ALERT "testlist : %d is added to the head\n", n);
}

static void testlist_add_tail(int n) {
    struct testlist_entry* e = kmalloc(sizeof(*e), GFP_KERNEL);
    e->n = n;
    list_add_tail(&e->list, &testlist);
    printk(KERN_ALERT "testlist : %d is added to the tail\n", n);
}

static void testlist_del_head(void) {
    struct testlist_entry* e = list_first_entry(&testlist, struct testlist_entry, list);
    int n = e->n;
    list_del(&e->list);
    kfree(e);
    printk(KERN_ALERT "testlist : %d is deleted from the head\n", n);
}

static void testlist_show(void) {
    struct testlist_entry* e;
    printk(KERN_ALERT "testlist : show contents\n");
    list_for_each_entry(e, &testlist, list) {
        printk(KERN_ALERT "\t%d\n", e->n);
    }
}

static int mod_init(void) {
    testlist_show();
    testlist_add(1);
    testlist_show();
    testlist_add(2);
    testlist_show();
    testlist_add(3);
    testlist_show();
    testlist_del_head();
    testlist_show();
    testlist_del_head();
    testlist_show();
    testlist_add_tail(4);
    testlist_show();
    testlist_del_head();
    testlist_show();
    testlist_del_head();
    testlist_show();

    return 0;
}

static void mod_exit(void) {
    // NOP.
}

module_init(mod_init);
module_exit(mod_exit);

