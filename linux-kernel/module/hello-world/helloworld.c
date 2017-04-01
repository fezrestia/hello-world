#include <linux/module.h>

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Aki.SHIMO. <fezrestia@gmail.com>");
MODULE_DESCRIPTION("Hello World Kernel Module");

static int mod_init(void) {
    printk(KERN_ALERT "Hello World !\n");
    return 0;
}

static void mod_exit(void) {
    // NOP.
}

module_init(mod_init);
module_exit(mod_exit);

